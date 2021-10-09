
from functools import partial
import pprint

import torch
from torch import nn

class ModelArchConfig():
    config = dict()
    act_dict = dict(
        sigmoid = nn.Sigmoid(),
        tanh    = nn.Tanh(),
        relu    = nn.ReLU(),
        elu     = nn.ELU(),
        lrelu   = nn.LeakyReLU(),
        silu    = nn.SiLU()
    )
    act_opts = list(act_dict.keys())
    skip_opts = [0, 1, 2]
    bn_layer_opts = ['none', 'conv', 'fc', 'all']
    bn_loc2act_opts = ['before', 'after']

    def __init__(self, act='relu', skip=0, bn_layer='all', bn_act='before'):
        # act: string for activation function
        # skip: number of skip connections
        # bn_layer: where/if to put batchnorm
        # bn_act: 'before'/'after' means BN before/after activation function
        self.configure_activation(act)
        self.configure_skip(skip)
        self.configure_batchnorm(bn_layer, bn_act)

    def configure_activation(self, act):
        self.config['act_fun'] = self.act_dict[act.lower()]

    def configure_skip(self, skip):
        self.config['skip'] = skip

    def configure_batchnorm(self, bn_layer, bn_act):
        bn_layer, bn_act = bn_layer.lower(), bn_act.lower()
        self.config['bn_conv'] = bn_layer == 'all' or bn_layer == 'conv'
        self.config['bn_fc'] = bn_layer == 'all' or bn_layer == 'conv'
        self.config['bn_b4_act'] = bn_act == 'before'

    def print_opts(self):
        pprint.pprint(dict(
            activation_functions        = self.act_opts,
            skip_connections            = self.skip_opts,
            batchnorm_layer_locs        = self.bn_layer_opts,
            batchnorm_locs_rel_to_act   = self.bn_loc2act_opts
        ))

class OptimAndSched():
    optim_dict = dict(
        sgd = torch.optim.SGD,
        adam = torch.optim.Adam
    )
    optim_opts = list(optim_dict.keys())

    sched_opts = ['const', 'step', 'c', 'reduce_on_plateau', 'anneal']
    sched_dict = dict(
        const = lambda x: None,
        step = torch.optim.lr_scheduler.StepLR,
        exp = torch.optim.lr_scheduler.ExponentialLR,
        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau,
        anneal = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    )

    def __init__(self, learning_rate=1e-3, optim='Adam', sched='const',
                 optim_args = dict(), sched_args = dict()):
        optim, sched = optim.lower(), sched.lower()
        if optim not in self.optim_opts or sched not in self.sched_opts:
            raise('Not allowed options')
        self.optim_fun = partial(self.optim_dict[optim], lr=learning_rate, **optim_args)
        self.sched_fun = partial(self.sched_dict[sched], **sched_args)

    def get_optim_and_sched(self, model):
        optimizer = self.optim_fun(model.parameters())
        scheduler = self.sched_fun(optimizer)
        return optimizer, scheduler

    def print_opts(self):
        pprint.pprint(dict(
            optimizers      = self.optim_opts,
            schedulers      = self.sched_opts
        ))