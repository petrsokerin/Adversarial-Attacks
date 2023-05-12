import os
import pickle
from typing import Dict
import shutil 

from .discrim_attack import HideAttackExp

import pandas as pd

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

    shutil.copyfile('config/experiment_run.yaml', path + f'/config_{dataset}_{model_id}_alpha={alpha}.yaml')

    aa_res_df.to_csv(path + f'/aa_res_{dataset}_{model_id}_alpha={alpha}.csv')
        
    with open(path + f'/rej_curves_dict_{dataset}_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
        pickle.dump(rej_curves_dict, file)


def save_train_disc(experiment, model_id, cfg):

    if "reg" in cfg['attack_type']:
        exp_name = f"{cfg['attack_type']}_eps={cfg['eps']}_alpha={cfg['alpha']}_nsteps={cfg['n_steps']}"
    else:
        exp_name = f"{cfg['attack_type']}_eps={cfg['eps']}_nsteps={cfg['n_steps']}"
        
    full_path = cfg['save_path'] + '/' + exp_name

    if not os.path.isdir(full_path):
        os.makedirs(full_path)
            
    with open(full_path+'/' + f"{model_id}.pickle", 'wb') as f:
        pickle.dump(experiment, f)


def load_disc_model(path='results/FordA/Regular/Discriminator_pickle', 
                    model_name='fgsm_attack_eps=0.03_nsteps=10',
                    device='cpu', 
                    model_id=0):
    path = fr'{path}/{model_name}/{model_id}.pickle'
    with open(path, 'rb') as f:
        experiment = pickle.load(f)
    disc_model = experiment.disc_model
    disc_model.to(device)
    disc_model.train(True)

    return disc_model