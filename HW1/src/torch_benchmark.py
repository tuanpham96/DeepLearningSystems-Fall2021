import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import time, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

from utils import *

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x).flatten(start_dim=1)

        x = self.d1(x)
        x = F.relu(x)

        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out

def get_device(device_type='GPU'):
    if device_type.upper() == 'GPU':
        if not torch.cuda.is_available():
            raise Exception('No cuda/GPU resources available for torch')

        device = torch.device("cuda:0")
        use_tpu = False
    elif device_type.upper() == 'TPU-1':
        try:
            print(os.environ['COLAB_TPU_ADDR'])
        except:
            raise Exception('No TPU resources available for torch')

        try:
            import torch_xla
        except:
            raise Exception('need to install XLA requirements for torch first')

        import torch_xla.core.xla_model as xm
        globals()['torch_xla'] = __import__('torch_xla')
        globals()['xm'] = xm
        device = xm.xla_device()
        use_tpu = True
    else:
        raise Exception('Invalid device type input for using torch. Only "GPU" or "TPU-1" (single core) supported')
    print(device)
    return device, use_tpu

def get_data_loaders(batch_size=32, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_accuracy(logit, target):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    batch_size = target.size()[0]
    return 100.0 * corrects/batch_size

@timerfunc
def train(model, device, train_loader, optimizer, criterion, use_tpu=False):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for cnt, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_tpu:
            xm.optimizer_step(optimizer, barrier=True)

        train_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels).item()

    cnt += 1
    return train_loss / cnt,  train_acc / cnt

@timerfunc
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            test_acc += get_accuracy(outputs, labels).item()
    cnt += 1
    return test_loss / cnt, test_acc / cnt

def run_each_model(model_id, device, criterion, config,
                    use_tpu = False, tqdm_args=dict(), print_perf=True):
    num_epoch = config['num_epoch']

    model = TorchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    bm_dict = dict(
        exp_begin   = datetime.now().strftime("%H:%M:%S"),
        model_id    = model_id,
        epoch       = [-1] * num_epoch,
        train_loss  = [-1] * num_epoch,
        train_acc   = [0] * num_epoch,
        train_time  = [0] * num_epoch,
        test_loss   = [-1] * num_epoch,
        test_acc    = [0] * num_epoch,
        test_time   = [0] * num_epoch
    )


    for i in tqdm(range(num_epoch), desc='ID={}'.format(model_id), **tqdm_args):
        bm_dict['epoch'][i] = i
        (bm_dict['train_loss'][i], bm_dict['train_acc'][i]), bm_dict['train_time'][i] \
            = train(model, device, train_loader, optimizer, criterion, use_tpu)
        (bm_dict['test_loss'][i], bm_dict['test_acc'][i]), bm_dict['test_time'][i] \
            = test(model, device, test_loader, criterion)
        if print_perf:
            print_epoch_progress(config, bm_dict, epoch=i)

    df = pd.DataFrame(bm_dict)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Benchmark')

    parser.add_argument('--device',         type=str,       default='GPU',      help='device, only "GPU" (default) or "TPU-1" accepted')
    parser.add_argument('--save-path',      type=str,       default='./',       help='path to save output (default: current)')
    parser.add_argument('--num-epoch',      type=int,       default=2,          help='number of epochs (default: 2)')
    parser.add_argument('--num-run',        type=int,       default=1,          help='number of runs (default: 1)')
    parser.add_argument('--learning-rate',  type=float,     default=0.001,      help='learning rate (default: 1e-3)')
    parser.add_argument('--batch-size',     type=int,       default=32,         help='batch size (default: 32)')
    parser.add_argument('--print-perf',     action='store_true',                help='print performance (default: off)')
    args = parser.parse_args()

    save_path = args.save_path
    print_perf = args.print_perf
    args = vars(args)

    # for debugging
    # config = dict(library = 'PyTorch', learning_rate = 0.001,
    #   num_epoch = 5,  batch_size = 32,  device = 'GPU', num_run = 3)

    config = dict(
        library = 'PyTorch',
        **args
    )
    del config['save_path']
    del config['print_perf']
    main_tqdm_args = dict(
        position=0, leave=False,
        desc="Main", ncols=50,
        colour='green'
    )

    each_tqdm_args = dict(
        position=1, leave=False, ncols=50
    )


    out_file = config_filename(config, save_path)
    device, use_tpu = get_device(config['device'])
    batch_size = config['batch_size']
    num_run = config['num_run']
    train_loader, test_loader = get_data_loaders(batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    dfs = []
    for i in tqdm(range(num_run), **main_tqdm_args):
        model_id = '%02d' %(i)
        dfs.append(run_each_model(model_id, device, criterion,
                                config, use_tpu, each_tqdm_args,
                                print_perf))
    df = pd.concat(dfs, ignore_index=True).assign(**config)

    df.to_csv(out_file)
    print('File saved at %s' %(out_file))
