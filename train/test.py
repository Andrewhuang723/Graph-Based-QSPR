import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.data import MolDataLoader
from data.utils import parse_type
from model.model import MPNN, FFNN
from typing import Callable, List
from tqdm import tqdm
import csv
import os
from pathlib import Path
from args import TrainArgs, PredictArgs
from .train import History

device = torch.device("cuda")

class save_predict:
    """
    Save prediction into save path by rows.
    """
    def __init__(self, save_path, name: str or list):
        self.save_path = save_path
        self.name = name
        self._detect()

    def _create_path(self):
        """
        Create save path for prediction if not exist.
        """
        row = [self.name] if isinstance(self.name, str) else self.name
        Path(self.save_path).touch()
        with open(self.save_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _detect(self):
        """
        (Remove the save path first if it exists)
        """
        if not os.path.exists(self.save_path):
            self._create_path()
        else:
            os.remove(self.save_path)
            self._create_path()

    def insert(self, pred: torch.Tensor):
        """
        Insert predictive data when it exists.
        """
        pred = pred.detach().cpu().numpy().tolist()
        with open(self.save_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(pred)


def predict(model: MPNN or FFNN,
            data_loader: MolDataLoader or DataLoader,
            loss_function: List[Callable],
            metric: List[Callable],
            args: TrainArgs or PredictArgs,
            saving: bool = False,
            return_logs: bool = False,):
    """
    data_loader: (graph, y) -> y could be multitask output
    loss_function: could be single loss function or multi-loss function (list)
    saving: save the prediction or not
    """
    model.eval()
    iter_count = 0
    loss_history = History(args)
    metric_history = History(args)

    save_predict_collections = {}
    collections = {}
    save_hidden_states = []
    if saving:
        for name, save_path in zip(args.task_names, args.save_path):
            save_predict_collections[name] = save_predict(save_path, name)

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False):
            batch_smiles = batch[0]
            batch_graph = batch[1]
            targets = batch[2:]
            batch_graph: dgl.DGLGraph

            if args.output_hidden_states:
                preds, hidden_states = model(batch_graph, batch_smiles, args.output_hidden_states)
                save_hidden_states.append(hidden_states)
            else:
                preds = model(batch_graph, batch_smiles, args.output_hidden_states)

            # multitask
            for pred, target, lf, met, dataset_type, task in zip(preds, targets, loss_function, metric,
                                                                 args.dataset_type, args.task_names):
                target = parse_type(target, dataset_type).to(device)
                if pred.shape == target.shape:
                    loss = lf(pred, target)
                    m = met(pred, target)

                    loss_history.log(task, loss)
                    metric_history.log(task, m)

                if saving:
                    if dataset_type == "classification" or dataset_type == "smiles":
                        pred_softmax = torch.log_softmax(pred, dim=-1)
                        _, pred = torch.max(pred_softmax, dim=-1)

                    save_predict_collections[task].insert(pred)

                if return_logs:
                    if collections.get(f"{task}_pred") is None:
                        collections[f"{task}_pred"] = pred
                        collections[f"{task}_target"] = target
                    else:
                        collections[f"{task}_pred"] = torch.cat([collections[f"{task}_pred"], pred], dim=0)
                        collections[f"{task}_target"] = torch.cat([collections[f"{task}_target"], target], dim=0)


            iter_count += 1

        if save_hidden_states:
            ht = torch.cat(save_hidden_states, dim=0)
            torch.save(ht, os.path.join("/".join(args.save_path[0].split("/")[:-1]), "hidden_states.pkl"))
            print("hidden states is saved")


    losses = [loss_history.results(task, iter_count) for task in args.task_names]
    metrics = [metric_history.results(task, iter_count) for task in args.task_names]
    if return_logs:
        return collections
    else:
        return losses, metrics
