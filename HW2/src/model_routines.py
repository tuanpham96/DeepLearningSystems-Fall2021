import random
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch import nn

from src.utils import *
from src.config_parsers import *
from src.model_construction import *

def set_seeds(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_accuracy(logit, target):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    batch_size = target.size()[0]
    return 100.0 * corrects/batch_size

@timerfunc
def train(model, device, data_loader, optimizer, criterion, scheduler=None, sched_type=None, epoch=0):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    iters = len(data_loader)

    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels).item()

        if sched_type == 'anneal':
            scheduler.step(epoch + cnt / iters)

    cnt += 1
    return train_loss / cnt,  train_acc / cnt

@timerfunc
def test(model, device, data_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for cnt, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            test_acc += get_accuracy(outputs, labels).item()
    cnt += 1
    return test_loss / cnt, test_acc / cnt

def step_scheduler(scheduler, sched_type, valid_loss):
    # if sched_type in ['const', 'anneal']: # do nothing
    if sched_type in ['exp', 'step']:
        scheduler.step()
    if sched_type == 'reduce_on_plateau':
        scheduler.step(valid_loss)

def run_each_model(config, device, data_loaders, tqdm_args=dict(), print_perf=True, print_keys=[]):
    # get individual data loader
    train_loader, valid_loader, test_loader = \
        data_loaders['train'],  data_loaders['validate'],  data_loaders['test']

    # for reproducibility
    set_seeds(config['seed'])

    # for easier access
    model_id = config['model_id']
    num_epoch = config['num_epoch']
    sched_type = config['sched'].lower()

    # create model
    model_arch_config = ModelArchConfig(config['act'], config['skip'], config['bn_layer'], config['bn_act'])

    model = ModelBM(**model_arch_config.config)
    model.apply(weight_init)
    model = model.to(device)

    # create optimizier and scheduler
    optim_sched_config = OptimAndSched(config['lr'], config['optim'], sched_type)
    optimizer, scheduler = optim_sched_config.get_optim_and_sched(model)

    criterion = nn.CrossEntropyLoss()
    bm_dict = init_benchmark_stats(model_id, num_epoch)

    # run through each epoch
    for i in tqdm(range(num_epoch), desc='ID={}'.format(model_id), **tqdm_args):
        epoch_num = i+1
        bm_dict['epoch'][i] = epoch_num

        # train
        (bm_dict['train_loss'][i], bm_dict['train_acc'][i]), bm_dict['train_time'][i] \
            = train(model, device, train_loader, optimizer, criterion, scheduler, sched_type, i)

        # validate
        (bm_dict['valid_loss'][i], bm_dict['valid_acc'][i]), bm_dict['valid_time'][i] \
            = test(model, device, valid_loader, criterion)

        # scheduler step
        valid_loss = bm_dict['valid_loss'][i]
        step_scheduler(scheduler, sched_type, valid_loss)

        # test at the end
        if epoch_num == num_epoch:
            (bm_dict['test_loss'], bm_dict['test_acc']), bm_dict['test_time'] \
                = test(model, device, test_loader, criterion)
        # print progress
        if print_perf:
            print_epoch_progress(config, bm_dict, epoch_num, print_keys)

    # save model
    torch.save(dict(
        config              = config,
        model_init_config   = model_arch_config.config,
        model_state_dict    = model.state_dict(),
        benchmark_stats     = bm_dict
    ), config['model_file'])

    # save stat
    df = pd.DataFrame(bm_dict).assign(**config)
    df.to_csv(config['stat_file'], index=False)

    return df

def test_runeachmodel(device, data_loaders):
    config = dict(
        exp_name = 'test',
        model_id = 1,
        act = 'relu',
        skip = 2,
        bn_layer = 'all',
        bn_act = 'before',
        optim = 'adam',
        sched = 'const',
        lr = 1e-3,
        num_epoch = EPOCHS,
        seed = SEED
    )

    config_filename(config)
    df = run_each_model(config, device, data_loaders, each_tqdm_args, print_keys='act')
    print(df)

def reload_model(model_file, device=None):
    model_data = torch.load(model_file)
    model = ModelBM(**model_data['model_init_config'])
    model.load_state_dict(model_data['model_state_dict'])
    if device is not None: model = model.to(device)
    return model