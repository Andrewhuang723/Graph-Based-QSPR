import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.modules import Module
from torch.utils.data import Subset, ConcatDataset
from pytorch_tools import EarlyStopping
from data.data import MolDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from typing import List
import os

device = torch.device("cuda")

def build_optimizer(model: Module, lr):
    return Adam(model.parameters(), lr=lr)


def early_stopping(patience: int):
    es = EarlyStopping(patience=patience)
    return es


def SelectiveSample(train_dataset: MolDataset, test_dataset: MolDataset, idx: List[int]):
    L = list(range(test_dataset.__len__()))
    n_idx = list(set(L).difference(idx))

    kth_dataset = Subset(test_dataset, idx)
    test_dataset = Subset(test_dataset, n_idx) # remove dataset
    train_dataset = ConcatDataset([train_dataset, kth_dataset]) # append dataset
    return train_dataset, test_dataset


def start_plot(figsize=(10, 8), style='whitegrid'):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0, 0])
    return ax


color_picker = ["navy", "teal", "darkorange", "gold", "k"]


def history_plot(history: dict, path, task_names: List[str]):
    ax = start_plot()
    for i, task in enumerate(task_names):
        VAL_LOSS = history[f"{task}_VAL_LOSS"]
        LOSS = history[f"{task}_LOSS"]

        ax.plot(LOSS, label='%s loss' % task, color=color_picker[i])
        ax.plot(VAL_LOSS, label='%s Validation loss' % task, color=color_picker[i], ls='--')
    ax.plot(history["LOSS"], label="Overall Training Loss", color="brown")
    ax.plot(history["VAL_LOSS"], label="Overall Validation Loss", color="brown", ls="--")
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title("Learning curve")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', frameon=True, shadow=True,  fontsize=15)
    plt.savefig(path)


def R2_plot(reconstruct_y_pred, reconstruct_y_test, prop_name_T, path):
    ax = start_plot(style='darkgrid')
    ax.scatter(reconstruct_y_test.reshape(-1), reconstruct_y_pred.reshape(-1), color='darkorange', edgecolor='navy',
               label=r'$R^2:\quad %.4f$' % r2_score(reconstruct_y_test, reconstruct_y_pred) + '\n' +
                     r'$MAE: \quad %.4f$' % mean_absolute_error(reconstruct_y_test, reconstruct_y_pred))
    ymin = min(np.min(reconstruct_y_test), np.min(reconstruct_y_pred)) - 0.1
    ymax = max(np.max(reconstruct_y_test), np.max(reconstruct_y_pred)) + 0.1
    lim = [ymin, ymax]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, c='brown', ls='--', label=r'$y=\hat y, identity$')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=15)
    plt.xlabel('TRUE %s' % prop_name_T, fontsize=20)
    plt.ylabel('PREDICTED %s' % prop_name_T, fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('%s Testing: %d' % (prop_name_T, len(reconstruct_y_test)), fontsize=20)
    plt.savefig(os.path.join(path, '%s R2.png' % prop_name_T))


def accuracy(yhat: torch.Tensor, y: torch.Tensor):
    batch_size, feat_size = y.shape
    yhat = yhat.round()
    identical = yhat == y
    identical = identical.float().sum()
    acc = identical / (feat_size * batch_size)
    return torch.tensor(acc).float()


def multi_accuracy(y_hat, y):
    batch_size, feat_size = y.shape
    y_hat_softmax = torch.log_softmax(y_hat, dim=-1)
    _, y_hat_tags = torch.max(y_hat_softmax, dim=-1)
    identical = y_hat_tags == y
    identical = identical.float().sum()
    acc = identical / (feat_size * batch_size)
    return torch.tensor(acc).float()


def cross_entropy(yhat: torch.Tensor, y: torch.Tensor):
    crossentropy = nn.CrossEntropyLoss().to(device)
    batch_size = yhat.shape[0]
    loss_sum = sum([crossentropy(s_hat, s) for s_hat, s in zip(yhat, y)])
    return loss_sum / batch_size