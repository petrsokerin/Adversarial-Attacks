import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


from typing import Dict, Any, Tuple, List, Union, Sequence, Callable


def fgsm_attack(loss_val, x, eps):
    grad_ = torch.autograd.grad(loss_val, x, retain_graph=True)[0]
    x_adv = x.data + eps * torch.sign(grad_)
    return x_adv


def fgsm_reg_attack(loss_val, x, eps, alpha):
    x_anchor = x[:, 1:-1]
    x_left = x[:, 2:]
    x_right = x[:, :-2]
    x_regular = (x_left + x_right) / 2
    loss_reg = torch.sum((x_anchor - x_regular.detach()) ** 2, dim=list(range(1, len(x.shape))))

    loss = loss_val - alpha * torch.mean(loss_reg)
    grad_ = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x.data + eps * (torch.sign(grad_))

    return x_adv

def fgsm_disc_attack(loss_val, x, eps, alpha, disc_model):
    torch.functional

    loss = loss_val - alpha * torch.mean(torch.log(F.sigmoid(disc_model(x))))
    grad_ = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x.data + eps * torch.sign(grad_)
    return x_adv


def build_df_aa_metrics(metric_dict: dict, eps: float):
    """
    Transform dict with metrics in pd.DataFrame

    :param metric_dict: dict key iter number and values list of metrics ACC, ROC AUC, PR AUC
    :param eps: eps param to add in result df
    :return: pd.DataFrame with metrics, number of iterations and eps

    """

    results_df = pd.DataFrame.from_dict(metric_dict, orient="index")
    results_df.set_axis(
        pd.Index(["ACC", "ROC AUC", "PR AUC"], name="metric"), axis=1, inplace=True
    )
    results_df.set_axis(
        pd.Index(results_df.index, name="n steps", ), axis=0, inplace=True,
    )

    results_df = results_df.reset_index()
    results_df['eps'] = eps
    return results_df


def calculate_metrics_class(y_true: np.array,
                            y_pred: np.array):
    # -> Tuple(float, float, float):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    pr = average_precision_score(y_true, y_pred)
    return acc, roc, pr


def calc_accuracy(y_true, y_pred, y_pred_adv):
    acc_val = np.mean((y_pred == y_true))
    acc_adv = np.mean((y_pred_adv == y_true))
    return acc_val, acc_adv


def req_grad(model: nn.Module, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.

    :param model: the model
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)


class IterGradAttack:
    def __init__(self, model, loader, attack_func, attack_params,
                 criterion, n_steps, train_mode=False):
        self.model = model
        self.loader = loader
        self.attack_func = attack_func
        self.attack_params = attack_params
        self.criterion = criterion
        self.n_steps = n_steps
        self.train_mode = train_mode

        self.dataset_class = loader.dataset.__class__
        self.device = next(model.parameters()).device
        self.batch_size = loader.batch_size

        self.logging = False

    def run_iterations(self):
        for iter_ in tqdm(range(self.n_steps)):

            if self.logging:
                x_adv, y_true, preds_original, preds_adv = self.run_one_iter()
                self.log_one_iter(iter_, y_true, preds_original, preds_adv)
            else:
                x_adv, y_true = self.run_one_iter()

            # rebuilding dataloader for new iteration
            it_dataset = self.dataset_class(x_adv, torch.tensor(y_true))
            self.loader = DataLoader(it_dataset, batch_size=self.batch_size)

        return x_adv, y_true

    def run_one_iter(self):

        self.model.train(self.train_mode)
        req_grad(self.model, state=False)  # detach all model's parameters

        all_y_true = torch.tensor([])  # logging y_true for rebuilding dataloader and calculation difference with preds
        x_adv_tensor = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader

        if self.logging:
            all_preds = []  # logging predictions original for calculation difference with data
            all_preds_adv = []  # logging predictions for calculation difference with data

        for x, y_true in self.loader:
            all_y_true = torch.cat((all_y_true, y_true.cpu().detach()), dim=0)

            x.grad = None
            x.requires_grad = True

            # prediction for original input
            x = x.to(self.device, non_blocking=True)
            y_true = y_true.to(self.device)
            y_pred = self.model(x)

            # attack for adv input
            loss_val = self.criterion(y_pred, y_true.reshape(-1, 1))
            x_adv = self.attack_func(loss_val, x, **self.attack_params)
            x_adv_tensor = torch.cat((x_adv_tensor, x_adv.cpu().detach()), dim=0)

            if self.logging:
                all_preds.extend(y_pred.cpu().detach().data.numpy())

                with torch.no_grad(): # prediction for adv input
                    y_pred_adv = self.model(x_adv)
                all_preds_adv.extend(y_pred_adv.cpu().detach().data.numpy())

        if self.logging:
            return x_adv_tensor.detach(), all_y_true.detach(), all_preds, all_preds_adv
        else:
            return x_adv_tensor.detach(), all_y_true.detach()


    def log_one_iter(self, iter_, y_true, preds_original, preds_adv):
        if self.multiclass:
            preds_flat_round = np.argmax(np.array(preds_original), axis=1).flatten()
            preds_adv_flat_round = np.argmax(np.array(preds_adv), axis=1).flatten()
            shape_diff = (1, 2)
        else:
            preds_flat_round = np.round_(np.array(preds_original)).flatten()
            preds_adv_flat_round = np.round_(np.array(preds_adv)).flatten()
            shape_diff = (1)

        y_true_flat = y_true.cpu().detach().numpy().flatten()

        # estimation
        if iter_ == 0:
            self.iter_broken_objs[preds_flat_round != y_true_flat] = iter_
            self.aa_res_dict[iter_] = self.metric_fun(y_true_flat, preds_flat_round)
            self.preds_iter_1 = np.array(preds_original)

        mask = (preds_adv_flat_round != y_true_flat) & (self.iter_broken_objs > iter_)
        self.iter_broken_objs[mask] = iter_ + 1
        self.rejection_dict['diff'][iter_ + 1] = np.sum((self.preds_iter_1 - np.array(preds_adv)) ** 2,
                                                        axis=shape_diff)
        self.rejection_dict['iter_broke'] = self.iter_broken_objs
        self.aa_res_dict[iter_ + 1] = self.metric_fun(y_true_flat, preds_adv_flat_round)


    def run_iterations_logging(self, metric_fun, n_objects, multiclass=False):

        self.metric_fun = metric_fun
        self.n_objects = n_objects
        self.multiclass = multiclass

        self.logging = True

        self.aa_res_dict = dict() #structure for saving decreasing of metrics
        self.rejection_dict = dict()  #structure for saving rejection curves params
        self.rejection_dict['diff'] = dict()
        self.iter_broken_objs = np.array([10 ** 7] * n_objects)

        self.run_iterations()

        return self.aa_res_dict, self.rejection_dict

# -----------------------------------------------------------------------------------------------------


def ifgsm_procedure(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    eps_params: Tuple[float, float, int],
                    n_steps: int,
                    metric_func=calculate_metrics_class,
                    n_objects=100,
                    train_mode=False,
                   ):
    aa_res_df = pd.DataFrame()

    rej_curves_dict = dict() # multilevel dict  eps -> diff and object
    # diff -> #n_iteration -> np.array difference between original prediction without attack and broken predictions
    # object -> np.array n_iter when wrong prediction

    eps_for_check = np.geomspace(*eps_params)

    for eps in tqdm(eps_for_check):
        print(f'*****************  EPS={eps}  ****************')

        attack_func = fgsm_attack
        attack_params = {'eps': eps}
        ifgsm_attack = IterGradAttack(model, loader, attack_func, attack_params,
                                      criterion, n_steps, train_mode=train_mode)
        aa_res_iter_dict, rej_curves_iter_dict = ifgsm_attack.run_iterations_logging(metric_func, n_objects, multiclass=False)

        rej_curves_dict[eps] = rej_curves_iter_dict
        aa_res_df = pd.concat([aa_res_df, build_df_aa_metrics(aa_res_iter_dict, eps)])

    return aa_res_df, rej_curves_dict

