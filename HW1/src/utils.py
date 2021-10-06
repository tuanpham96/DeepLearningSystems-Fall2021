import time, os
import pandas as pd 
import numpy as np 
from datetime import datetime
import argparse

def config_filename(config, mainpath='./data'):
    filename = '{library}-{device}.csv'.format(**config)
    filename = os.path.join(mainpath, filename)
    config['filename'] = filename
    return filename

def timerfunc(func):
    def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        return value, runtime
    return function_timer

def print_epoch_progress(config, stats, epoch=-1):
    if epoch == -1: # print the end by default 
        epoch = len(stats['epoch'])-1

    total_time = (sum(stats['train_time']) + sum(stats['test_time']))/60

    stats_at_epoch = {k: v[epoch] for k,v in stats.items() if isinstance(v, list)}
    
    epoch_time = stats_at_epoch['train_time'] + stats_at_epoch['test_time']
    msg = '\n{device}-{library} @ epoch {curr_epoch}/{num_epoch}'\
    ' took {epoch_time:.1f} secs (total {total_time:.2f} mins elapsed)\n' \
        '\t + TRAIN: ACC = {train_acc:.2f}%\t| LOSS = {train_loss:.4f} | TIME = {train_time:.1f} s \n' \
        '\t + TEST:  ACC = {test_acc:.2f}%\t| LOSS = {test_loss:.4f} | TIME = {test_time:.1f} s'

    print(msg.format(curr_epoch=epoch+1, epoch_time=epoch_time, total_time=total_time,
                        **config, **stats_at_epoch))