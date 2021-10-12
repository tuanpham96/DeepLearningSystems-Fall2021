import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

from tqdm.notebook import tqdm
import yaml

from src.utils import *
from src.data_callers import *
from src.config_parsers import *
from src.model_construction import *
from src.model_routines import *

def run_experiment(exp_config, device, data_loaders,
                   out_path='output', tqdm_args=main_tqdm_args,
                   restart_exp = False):
    exp_name = exp_config['name']

    # save config first
    exp_conf_file = os.path.join(out_path, exp_name + '_exp_conf.yml')
    if restart_exp:
        with open(exp_conf_file, 'w') as file:
            yaml.dump(exp_config, file, default_flow_style=False)

    # param combinations
    exp_param_combs = list(ParameterGrid(exp_config['param']))
    num_combs = len(exp_param_combs)

    dfs = []

    for i, prm in tqdm(enumerate(exp_param_combs), total=num_combs, **tqdm_args):
        # create each model config
        config = dict(
            exp_name = exp_name,
            model_id = i,
            **exp_config['const'],
            **prm)
        config_filename(config)

        print_keys = list(prm.keys())

        if os.path.exists(config['model_file']) and os.path.exists(config['stat_file']) and not restart_exp:
            # check if already run
            df = pd.read_csv(config['stat_file'])
        else:
            # run and (optionally) report
            df = run_each_model(
                config,
                device,
                data_loaders,
                each_tqdm_args,
                print_perf=True,
                print_keys=print_keys
            )

        dfs.append(df)

    # save stat of exp
    df = pd.concat(dfs, ignore_index=True)
    exp_stat_file = os.path.join(out_path, exp_name + '_exp_stat.csv')
    df.to_csv(exp_stat_file, index=False)

    return df
