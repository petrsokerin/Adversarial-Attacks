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
from utils.attacks import fgsm_disc_attack, fgsm_attack, fgsm_reg_attack

def load_disc_model(model_name='fgsm_attack_eps=0.03_nsteps=10', device='cpu'):
    with open(fr'results/Ford_A/Regular/Discriminator_pickle/{model_name}/0.pickle', 'rb') as f:
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

    attack_func = fgsm_reg_attack
    #attack_func = fgsm_attack

    disc_model = load_disc_model(device=device)

    for alpha in tqdm([100]):
        attack_params = {'alpha':alpha}
        #attack_params = dict()

        for model_id in range(1):

            model = LSTM_net(hidden_dim = 50, n_layers = 1, output_dim = 1, dropout=0.0).to(device)
            model_path = model_folder + f'model_{0}_FordA.pth'
            model.load_state_dict(copy.deepcopy(torch.load(model_path)))

            aa_res_df, rej_curves_dict = ifgsm_procedure(model=model, loader=test_loader, criterion=criterion,
                                                        attack_func=attack_func, attack_params=attack_params,
                                                        eps_params=eps_params, n_steps=n_iters,
                                                        n_objects=n_objects, train_mode=train_mode,
                                                        disc_model=disc_model)

            aa_res_df.to_csv(f'results/Ford_A/Regular/aa_res_Ford_A_{model_id}_alpha={alpha}.csv')
            with open(f'results/Ford_A/Regular/rej_curves_dict_Ford_A_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
                pickle.dump(rej_curves_dict, file)


if __name__=='__main__':
    main()



