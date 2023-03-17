from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader

from .attacks import IterGradAttack

class HideAttackExp:
    def __init__(self, attack_model, train_loader, test_loader, attack_train_params, 
                 attack_test_params, discriminator_model, disc_train_params, multiclass=False):
        
        self.attack_loaders = {'train': train_loader,
                       'test': test_loader}
        self.attack_train = {'train':IterGradAttack(attack_model, train_loader, **attack_train_params),
                             'test': IterGradAttack(attack_model, test_loader, **attack_test_params)}
        self.disc_loaders = dict()
        
        self.attack_train_params = attack_train_params
        self.attack_test_params = attack_test_params
        
        
        self.eps = attack_train_params['attack_params']['eps']
        self.alpha = None
        if 'alpha' in attack_train_params['attack_params']:
            self.alpha = attack_train_params['attack_params']['alpha']
        self.n_steps = attack_train_params['n_steps']
        self.multiclass = multiclass

        
        self.attack_model = attack_model
        self.disc_model = discriminator_model
        
        self.disc_criterion = torch.nn.BCELoss()
        self.disc_n_epoch = disc_train_params['n_epoch']
        self.disc_batch_size = 64
        self.disc_optimizer = disc_train_params['optimizer']
        if 'scheduler' in disc_train_params.keys():
            self.disc_scheduler = disc_train_params['scheduler']
        else:
            self.disc_scheduler = None

        self.attack_device = next(attack_model.parameters()).device
        self.disc_device = next(discriminator_model.parameters()).device
        
    def run(self):
        print("Generating adv data")
        self.get_disc_dataloaders()
        print("Train discriminator")
        self.train_discriminator()
    
    def _generate_adv_data(self, mode='train', batch_size=64):
        
        self.disc_batch_size = batch_size
        dataset_class = self.attack_train[mode].dataset_class

        X_adv, y_adv = self.attack_train[mode].run_iterations()
        X_orig = torch.tensor(self.attack_loaders[mode].dataset.X)
        X_adv = X_adv.squeeze(-1)
    
        disc_labels_zeros = torch.zeros((len(X_orig), 1)) #True label class
        disc_labels_ones = torch.ones(y_adv.shape) #True label class
        
        new_x = torch.concat([X_orig, X_adv], dim=0)
        new_y = torch.concat([disc_labels_zeros, disc_labels_ones], dim=0)

        suffle_status = mode == 'train'
        disc_loader = DataLoader(dataset_class(new_x, new_y), batch_size=batch_size, shuffle=suffle_status)
        self.disc_loaders[mode] = disc_loader
        return disc_loader
    
    def get_disc_dataloaders(self):
        self._generate_adv_data('train')
        self._generate_adv_data('test')
        
    def _logging_train_disc(self, data, mode='train'):
        
        for metric in self.dict_logging[mode].keys():
            self.dict_logging[mode][metric].append(data[metric])
    
    def train_discriminator(self):
        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'balance']
        self.dict_logging = {'train': {metric:[] for metric in metric_names},
                       'test': {metric:[] for metric in metric_names}}

        for epoch in tqdm(range(self.disc_n_epoch)):
            train_metrics_epoch = self._train_step()
            train_metrics_epoch = {met_name:met_val for met_name, met_val
                                   in zip(metric_names, train_metrics_epoch)}
            self._logging_train_disc(train_metrics_epoch, mode='train')
            
            test_metrics_epoch = self._valid_step() 
            test_metrics_epoch = {met_name:met_val for met_name, met_val 
                       in zip(metric_names, test_metrics_epoch)}
            self._logging_train_disc(test_metrics_epoch, mode='test')
                    
    
    def _train_step(self):
        losses, n_batches = 0, 0
        
        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])
        
        self.disc_model.train(True)
        for x, labels in self.disc_loaders['train']:

            self.disc_optimizer.zero_grad()
            x = x.to(self.disc_device)
            labels = labels.reshape(-1, 1).to(self.disc_device)

            y_out = self.disc_model(x)
            
            loss = self.disc_criterion(y_out, labels) 

            loss.backward()     
            self.disc_optimizer.step()
            losses += loss
            n_batches += 1
            
            if self.multiclass:
                y_pred = torch.argmax(y_out, axis=1)
            else:
                y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = losses / n_batches

        if self.disc_scheduler:
            self.disc_scheduler.step()
            
        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        acc, pr, rec, f1 = self.calculate_metrics(y_all_true, y_all_pred)
        balance = np.sum(y_all_pred) / len(y_all_pred)
        return mean_loss, acc, pr, rec, f1, balance


    def _valid_step(self):
        
        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])
        
        losses, n_batches = 0, 0
        self.disc_model.eval()    
        for x, labels in self.disc_loaders['test']:
            with torch.no_grad():
                x = x.to(self.disc_device)
                labels = labels.reshape(-1, 1).to(self.disc_device)

                y_out = self.disc_model(x)
                loss = self.disc_criterion(y_out, labels)
                losses += loss
                n_batches += 1
                
                if self.multiclass:
                    y_pred = torch.argmax(y_out, axis=1)
                else:
                    y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = losses / n_batches
        
        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        acc, pr, rec, f1 = self.calculate_metrics(y_all_true, y_all_pred)
        balance = np.sum(y_all_pred) / len(y_all_pred)
        return mean_loss, acc, pr, rec, f1, balance


    def calculate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pr = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, pr, rec, f1
