import time, os, glob
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# to be perused by mains of others
main_tqdm_args = dict(
    position=0, leave=False,
    desc="Main", ncols=50,
    colour='green'
)

each_tqdm_args = dict(
    position=1, leave=False, ncols=50
)


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

def init_benchmark_stats(model_id, num_epoch):
    return dict(
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

def write_csv(df, outfile, overwrite=True):
    if glob.glob(outfile):
        if overwrite:
            print('File {} already exists. Will OVERWRITE file'.format(outfile))
        else:
            # for some reason a field cannot concatenate due to type BS
            print('File {} already exists. Will CONCATENATE file'.format(outfile))
            tmp_dir = './tmp/'
            if not glob.glob(tmp_dir):
                os.mkdir(tmp_dir)
            tmp_file = os.path.join(tmp_dir, 'tmp.csv')
            df.to_csv(tmp_file, index=False)
            df_old = pd.read_csv(outfile)
            max_old = df_old.model_id.max()
            df_new = pd.read_csv(tmp_file)
            df_new.model_id = df_new.model_id + max_old + 1
            df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(outfile, index=False)

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

def common_parser(description):
    # not parsing device, that is done separately
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--save-path',      type=str,       default='./',       help='path to save output (default: current)')
    parser.add_argument('--num-epoch',      type=int,       default=2,          help='number of epochs (default: 2)')
    parser.add_argument('--num-run',        type=int,       default=1,          help='number of runs (default: 1)')
    parser.add_argument('--learning-rate',  type=float,     default=0.001,      help='learning rate (default: 1e-3)')
    parser.add_argument('--batch-size',     type=int,       default=32,         help='batch size (default: 32)')
    parser.add_argument('--print-perf',     action='store_true',                help='print performance (default: off)')
    parser.add_argument('--overwrite',      action='store_true',                help='overwrite output file (default: off)')

    return parser
