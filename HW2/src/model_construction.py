import torch
import torch.nn.functional as F
from torch import nn

def make_fclayer(inp_dim, out_dim, act_fun = nn.ReLU(), bn_on = True, bn_b4_act = True):
    layer = [nn.Linear(inp_dim, out_dim)]
    bn = nn.BatchNorm1d(out_dim)
    if bn_on:
        if bn_b4_act:
            layer.extend([bn, act_fun])
        else:
            layer.extend([act_fun, bn])

    return nn.Sequential(*layer)

class ResidualLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, act_fun, bn_on, bn_b4_act, skip_inp=None, npoolb4skip = 0):
        super(ResidualLayer, self).__init__()
        self.bn_on = bn_on
        self.bn_b4_act = bn_b4_act
        self.skip_on = skip_inp is not None
        self.npoolb4skip = npoolb4skip

        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(skip_inp, out_dim, kernel_size=1, bias=False) if skip_inp else None
        self.bn = nn.BatchNorm2d(out_dim) if bn_on else None
        self.act = act_fun
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, residual = None):
        x = self.conv(x)

        if self.bn_on and self.bn_b4_act:
            x = self.bn(x)

        if self.skip_on and (residual is not None):
            for i in range(self.npoolb4skip):
                residual = self.pool(residual)
            x = x + self.skip(residual)

        x = self.act(x)

        if self.bn_on and not self.bn_b4_act:
            x = self.bn(x)

        x = self.pool(x)
        return x

class ModelBM(nn.Module):
    def __init__(self, act_fun = nn.ReLU(), skip = 0,
                 bn_conv = True, bn_fc = True, bn_b4_act = True):
        super(ModelBM, self).__init__()

        self.skip = skip
        npoolb4skip = [0, 0, 0]
        if skip == 0:
            skip_conv_dims = [None, None, None]
        elif skip == 1:
            skip_conv_dims = [3, 16, 32]
        else:
            skip_conv_dims = [None, 3, None]
            npoolb4skip[1] = 1

        conv_args = [act_fun, bn_conv, bn_b4_act]
        self.conv1 = ResidualLayer(3, 16, *conv_args, skip_conv_dims[0], npoolb4skip[0])
        self.conv2 = ResidualLayer(16, 32, *conv_args, skip_conv_dims[1], npoolb4skip[1])
        self.conv3 = ResidualLayer(32, 64, *conv_args, skip_conv_dims[2], npoolb4skip[2])

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = make_fclayer(4*4*64, 500, act_fun, bn_fc, bn_b4_act)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(500, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        res = x if self.skip > 0 else None
        x = self.conv1(x, res)

        if self.skip == 1: res = x
        x = self.conv2(x, res)

        if self.skip == 1: res = x
        x = self.conv3(x, res)

        x = self.flatten(x)

        x = self.fc1(x)
        out = self.fc2(x)
        return out

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
