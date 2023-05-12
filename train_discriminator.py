import copy

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch

from tqdm.notebook import tqdm
from utils.attacks import (fgsm_disc_attack, fgsm_attack, fgsm_reg_attack, 
simba_binary, simba_binary_reg, simba_binary_disc_reg)
from utils.discrim_attack import HideAttackExp
from utils.data import load_Ford_A, transform_Ford_A, build_dataloaders
from utils.config import get_attack
from utils.utils import save_train_disc

@hydra.main(config_path='config', config_name='train_disc', version_base=None)
def main(cfg: DictConfig):

    if cfg['dataset'] == 'FordA':
        X_train, X_test, y_train, y_test = load_Ford_A()
        X_train, X_test, y_train, y_test = transform_Ford_A(X_train, X_test, y_train, y_test)
        train_loader, test_loader = build_dataloaders(X_train, X_test, y_train, y_test)

    device= torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')

    for model_id in tqdm(cfg['model_ids']):

        attack_model = instantiate(cfg.attack_model).to(device)
        model_path = cfg['model_folder'] + f'model_{model_id}_FordA.pth'
        attack_model.load_state_dict(copy.deepcopy(torch.load(model_path)))
        
        attack_params = dict()

        attack_func = get_attack(cfg['attack_type'])

        if 'reg' in cfg['attack_type'] :
            attack_params['alpha'] = cfg['alpha']

        elif 'disc' in cfg['attack_type']:
            attack_params['alpha'] = cfg['alpha']

        attack_train_params = {'attack_func':attack_func, 
                        'attack_params':attack_params, 
                        'criterion':torch.nn.BCELoss(), 
                        'n_steps':cfg['n_steps'],
                        'train_mode': True}
        attack_test_params = attack_train_params

        discriminator_model = instantiate(cfg.disc_model).to(device)
        optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=cfg['lr'])
        disc_train_params = {'n_epoch': cfg['n_epochs'],
                            'optimizer': optimizer,
                            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, cfg['step_lr'], gamma=cfg['gamma'])}      

        experiment = HideAttackExp(attack_model, train_loader, test_loader, attack_train_params, 
                                attack_test_params, discriminator_model, disc_train_params)
        experiment.run()

        save_train_disc(experiment, model_id, cfg)

if __name__=='__main__':
    main()

