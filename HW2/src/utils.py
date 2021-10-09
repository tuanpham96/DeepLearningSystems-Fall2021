import os, time, glob
from datetime import datetime

# tqdm args to be shared
main_tqdm_args = dict(
    position=0, leave=True,
    desc="Main", ncols=500,
    colour='green'
)

each_tqdm_args = dict(
    position=1, leave=False, ncols=500
)

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
        valid_loss  = [-1] * num_epoch,
        valid_acc   = [0] * num_epoch,
        valid_time  = [0] * num_epoch,
        test_loss   = -1,
        test_acc    = 0,
        test_time   = 0,
    )

def config_filename(config, mainpath='output'):
    file_prefix = '{exp_name}_model-{model_id}'.format(**config)
    file_prefix = os.path.join(mainpath, file_prefix)
    config['stat_file'] = file_prefix +'_stat.csv'
    config['model_file'] = file_prefix +'_model.pt'


def print_epoch_progress(config, stats, epoch=-1, print_keys=[]):
    max_epoch = len(stats['epoch'])
    if epoch == -1: # print the end by default
        epoch = max_epoch

    total_time = sum(stats['train_time']) + sum(stats['valid_time']) + stats['test_time']
    total_time /= 60.0

    stats_at_epoch = {k: v[epoch-1] for k,v in stats.items() if isinstance(v, list)}
    epoch_time = stats_at_epoch['train_time'] + stats_at_epoch['valid_time']

    if not isinstance(print_keys, list): print_keys = [print_keys]
    print_var = '-'.join(['{}']*len(print_keys)).format(*[config[k] for k in print_keys])
    print_var = ' (vary: %s) ' %(print_var) if len(print_var) > 0 else ''

    msg = '{exp_name}-{model_id}{print_var}\t || ' \
        'EPOCH: {curr_epoch:02d}/{num_epoch:02d} | iter = {epoch_time:.1f}s | total = {total_time:.2f}m || ' \
        'TRAIN: acc = {train_acc:.1f} | loss = {train_loss:.3f} | time = {train_time:.1f}s || ' \
        'VALID: acc = {valid_acc:.1f} | loss = {valid_loss:.3f} | time = {valid_time:.1f}s || '

    stats_at_end = {}
    if epoch == max_epoch:
        msg += '\n\t\t ==> TEST: acc = {test_acc:.1f} | loss = {test_loss:.3f} | time = {test_time:.1f}s '
        stats_at_end = {k:v for k, v in stats.items() if 'test' in k}

    print(msg.format(print_var=print_var,
                     curr_epoch=epoch,
                     epoch_time=epoch_time,
                     total_time=total_time,
                     **config,
                     **stats_at_epoch,
                     **stats_at_end))
