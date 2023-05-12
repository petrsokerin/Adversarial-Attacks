import pickle
import copy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from utils.data import load_Ford_A, transform_Ford_A, MyDataset
from models.models import LSTM_net

from utils.attacks import ifgsm_procedure
from utils.discrim_attack import HideAttackExp
from utils.attacks import (fgsm_disc_attack, fgsm_attack, fgsm_reg_attack, 
simba_binary, simba_binary_reg, simba_binary_disc_reg)

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


def main():
    X_train, X_test, y_train, y_test = load_Ford_A()
    X_train, X_test, y_train, y_test = transform_Ford_A(X_train, X_test, y_train, y_test)

    BS = 64    
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=BS, shuffle=False)

    model_folder = 'checkpoints/Ford_A/'

    n_iters = 50
    eps_params = (1e-3, 1e0, 5)
    criterion = torch.nn.BCELoss()
    n_objects = y_test.shape[0]
    device= torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_mode=True

    #attack_func = fgsm_disc_attack #fgsm_disc_attack
    attack_func = fgsm_attack

    path = 'results/Ford_A/Regular/Discriminator_pickle'
    disc_model_reg1 = load_disc_model(model_id=0, path=path, model_name='fgsm_reg_attack_eps=0.03_alpha=0.01_nsteps=10', device=device)
    disc_model_reg2 = load_disc_model(model_id=0, path=path, model_name='fgsm_reg_attack_eps=0.03_alpha=0.1_nsteps=10', device=device)
    disc_model_reg3 = load_disc_model(model_id=1, path=path, model_name='fgsm_reg_attack_eps=0.03_alpha=0.01_nsteps=10', device=device)
    disc_model_reg4 = load_disc_model(model_id=1, path=path, model_name='fgsm_reg_attack_eps=0.03_alpha=0.1_nsteps=10', device=device)
    disc_model_check = load_disc_model(model_id=0, path=path, model_name='fgsm_attack_eps=0.03_nsteps=10', device=device)

    # path = 'results/Ford_A/SimBA/Discriminator_pickle'
    # disc_model_reg1 = load_disc_model(model_id=0, path=path, model_name='simba_binary_eps=0.1_nsteps=20',  device=device)
    # disc_model_reg2 = load_disc_model(model_id=1, path=path, model_name='simba_binary_eps=0.3_nsteps=20',  device=device)
    # disc_model_check = load_disc_model(model_id=0, path=path, model_name='simba_binary_eps=0.3_nsteps=20',  device=device)

    for alpha in tqdm([0.001, 0.01, 0.1, 1, 10, 100,]):
        #attack_params = {'alpha':alpha, 'disc_models': [disc_model_reg1, disc_model_reg2]} #, disc_model_reg3, disc_model_reg4]}
        attack_params = {'alpha': alpha}
        attack_params = dict()

        for model_id in range(1, 2):

            model = LSTM_net(hidden_dim = 50, n_layers = 1, output_dim = 1, dropout=0.0).to(device)
            model_path = model_folder + f'model_{model_id}_FordA.pth'
            model.load_state_dict(copy.deepcopy(torch.load(model_path)))

            aa_res_df, rej_curves_dict = ifgsm_procedure(model=model, loader=test_loader, criterion=criterion,
                                                        attack_func=attack_func, attack_params=attack_params,
                                                        eps_params=eps_params, n_steps=n_iters,
                                                        n_objects=n_objects, train_mode=train_mode,
                                                        disc_model=disc_model_check)

            aa_res_df.to_csv(f'results/Ford_A/Regular_Disc/Diff_many_4/aa_res_Ford_A_{model_id}_alpha={alpha}.csv')
            with open(f'results/Ford_A/Regular_Disc/Diff_many_4/rej_curves_dict_Ford_A_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
                pickle.dump(rej_curves_dict, file)

            # aa_res_df.to_csv(f'results/Ford_A/SimBA/Regular_Disc_diff_many_2/aa_res_Ford_A_{model_id}_alpha={alpha}.csv')
            # with open(f'results/Ford_A/SimBA/Regular_Disc_diff_many_2/rej_curves_dict_Ford_A_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
            #     pickle.dump(rej_curves_dict, file)


if __name__=='__main__':
    main()



