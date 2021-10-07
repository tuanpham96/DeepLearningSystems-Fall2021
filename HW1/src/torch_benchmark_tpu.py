import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from torch_benchmark import TorchModel, get_accuracy

import time, os, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

from utils import *



def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_ds, test_ds

def get_data_loaders(batch_size=32, num_workers=2):
    train_ds, test_ds = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader

def run_training_fn(WRAPPED_MODEL, model_id, config):

    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epoch = config['num_epoch']

    train_loader, test_loader = get_data_loaders(batch_size)

    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    def train_loop_fn(loader):
        model.train()
        train_loss = 0
        train_acc = 0
        for cnt, (data, target) in enumerate(loader):
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            train_loss += loss.item()
            train_acc += get_accuracy(output, target)

        cnt += 1
        train_loss /= cnt
        train_acc /= cnt
        return train_loss , train_acc.item()

    def test_loop_fn(loader):
        model.eval()
        test_loss = 0
        test_acc = 0
        total_samples = 0

        for cnt, (data, target) in enumerate(loader):
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            test_acc += get_accuracy(output, target)

        cnt += 1
        test_loss /= cnt
        test_acc /= cnt

        return test_loss, test_acc.item()

    bm_stats = init_benchmark_stats(model_id, num_epoch)

    for i in range(num_epoch):
        bm_stats['epoch'] = i
        para_loader = pl.ParallelLoader(train_loader, [device])

        t0 = time.time()
        bm_stats['train_loss'][i], bm_stats['train_acc'][i] = \
            train_loop_fn(para_loader.per_device_loader(device))

        para_loader = pl.ParallelLoader(test_loader, [device])

        t1 = time.time()

        bm_stats['test_loss'][i], bm_stats['test_acc'][i] =  \
            test_loop_fn(para_loader.per_device_loader(device))

        t2 = time.time()

        bm_stats['train_time'][i] = t1 - t0
        bm_stats['test_time'][i] = t2 - t1

    return bm_stats

def run_each_model(model_id, config, num_cores=8):
    tmp_dir = './tmp'
    if not glob.glob(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_file = os.path.join(tmp_dir, 'tmptpu.csv')

    num_epoch = config['num_epoch']
    WRAPPED_MODEL = xmp.MpModelWrapper(TorchModel())

    def _mp_fn(rank, config):
        torch.set_default_tensor_type('torch.FloatTensor')
        bm = run_training_fn(WRAPPED_MODEL, model_id, config)
        df = pd.DataFrame(bm)
        if rank == 0:
            write_csv(df, tmp_file)

    xmp.spawn(_mp_fn, args=(config,), nprocs=num_cores, start_method='fork')

    df = pd.read_csv(tmp_file)
    return df

if __name__ == '__main__':

    parser = common_parser('PyTorch Benchmark TPU multicore')
    parser.add_argument('--num-tpu-cores',      type=int,       default=8,          help='number of TPU cores (default: 8)')

    args = parser.parse_args()

    save_path = args.save_path
    overwrite_file = args.overwrite
    print_perf = args.print_perf
    num_cores = args.num_tpu_cores

    args = vars(args)
    config = dict(
        library = 'PyTorch',
        device = 'TPU',
        **args
    )
    del config['save_path']
    del config['print_perf']
    del config['num_tpu_cores']

    out_file = config_filename(config, save_path)

    batch_size = config['batch_size']
    num_run = config['num_run']

    SERIAL_EXEC = xmp.MpSerialExecutor()

    dfs = []

    for i in tqdm(range(num_run), **main_tqdm_args):
        model_id = '%02d' %(i)
        dfs.append(run_each_model(model_id, config, num_cores))
    df = pd.concat(dfs, ignore_index=True).assign(**config)

    write_csv(df, out_file, overwrite_file)

