import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from data.data import MolDataLoader
from data.utils import parse_type
from model.model import MPNN, FFNN, Set2Set
from typing import Callable, List
from tqdm import tqdm
from args import TrainArgs

device = torch.device("cuda")

class History:
    def __init__(self, args: TrainArgs):
        self.task_names = args.task_names
        self.history = self._initiated()

    def _initiated(self):
        history = {}
        for n in self.task_names:
            history[n] = 0
        return history

    def log(self, task, value):
        self.history[task] += value

    def results(self, task, count):
        return self.history[task] / count

    def parse_results(self, count):
        return


def train(model: MPNN or FFNN or Set2Set,
          data_loader: MolDataLoader or DataLoader,
          loss_function: List[Callable],
          metric: List[Callable],
          optimizer: Optimizer,
          args: TrainArgs):
    """
    data_loader: (graph, y) -> y could be multitask output
    loss_function: could be single loss function or multi-loss function (list)
    """
    model.train()
    iter_count = 0
    loss_history = History(args)
    metric_history = History(args)

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        batch_smiles = batch[0]
        batch_graph = batch[1]
        targets = batch[2:]

        batch_graph: dgl.DGLGraph
        preds = model(batch_graph, batch_smiles)

        # multitask
        loss_grad = 0
        for pred, target, lf, met, dataset_type, task in zip(preds, targets, loss_function, metric, args.dataset_type, args.task_names):
            target = parse_type(target, dataset_type).to(device)
            loss = lf(pred, target)
            loss_grad += loss # for backward propagation
            m = met(pred, target)

            loss_history.log(task, loss)
            metric_history.log(task, m)

        optimizer.zero_grad()
        loss_grad.backward()
        optimizer.step()

        iter_count += 1

    losses = [loss_history.results(task, iter_count) for task in args.task_names]
    metrics = [metric_history.results(task, iter_count) for task in args.task_names]
    return losses, metrics


