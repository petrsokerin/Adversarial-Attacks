import copy
import os
import pickle
from typing import Dict
import shutil 

from .discrim_training import HideAttackExp
from utils.data import MyDataset

import pandas as pd
import numpy as np
import torch
import random

def save_experiment(
        aa_res_df: pd.DataFrame, 
        rej_curves_dict: Dict,
        path: str, 
        dataset: str, 
        model_id: int, 
        alpha: float
    ) -> None:

    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copyfile('config/attack_run_config.yaml', path + f'/config_{dataset}_{model_id}_alpha={alpha}.yaml')

    aa_res_df.to_csv(path + f'/aa_res_{dataset}_{model_id}_alpha={alpha}.csv')
        
    with open(path + f'/rej_curves_dict_{dataset}_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
        pickle.dump(rej_curves_dict, file)


def save_train_disc(experiment, model_id, cfg):

    if 'prefix' not in cfg:
        cfg['prefix'] = ''

    if "reg" or 'disc' in cfg['attack_type']:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_alpha={cfg['alpha']}_nsteps={cfg['n_iterations']}"
    else:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_nsteps={cfg['n_iterations']}"
        
    full_path = cfg['save_path'] + '/' + exp_name

    if not os.path.isdir(full_path):
        os.makedirs(full_path)
            
    # with open(full_path+'/' + f"{model_id}.pickle", 'wb') as f:
    #     pickle.dump(experiment, f)

    model_weights_name = full_path + '/' + f"{model_id}.pt"
    torch.save(experiment.disc_model.state_dict(), model_weights_name)

    logs_name =  full_path+'/' + f"{model_id}_logs.pickle"
    with open(logs_name, 'wb') as f:
        pickle.dump(experiment.dict_logging, f)

    shutil.copyfile('config/train_disc_config.yaml', full_path+'/' + f"{model_id}_config.yaml")


def save_train_classifier(model, save_path, model_name):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    full_path = save_path + '/' + model_name
    torch.save(model.state_dict(), full_path) 


def load_disc_model(
        disc_model,
        path='results/FordA/Regular/Discriminator_pickle', 
        model_name='fgsm_attack_eps=0.03_nsteps=10',
        device='cpu', 
        model_id=0
        ):
    path = fr'{path}/{model_name}/{model_id}.pt'

    disc_model = copy.deepcopy(disc_model)
    disc_model.load_state_dict(torch.load(path))
    disc_model.to(device)
    disc_model.train(True)

    return disc_model

def fix_seed(seed: int) -> None:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
