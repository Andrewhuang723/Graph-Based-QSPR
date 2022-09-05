import dgl
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Tuple, List
import torch
import pandas as pd
from args import TrainArgs, PredictArgs


def get_all_index(d: List, value):
    idx_list = []
    for i in range(len(d)):
        if d[i] == value:
            idx_list.append(i)
    return idx_list


class Preprocess:
    def __init__(self, path: str):
        """
        path: the csv file storing the regression property mean and std.
        """
        self.path = path
        self.mean, self.std = self._read()

    def _read(self):
        if self.path is None:
            return 0, 1
        scaled = pd.read_csv(self.path)
        mean = scaled["mean"].values
        std = scaled["std"].values
        return mean, std

    def scaler(self, data: torch.Tensor):
        scale = (data - self.mean) / self.std
        return scale

    def inverse_scaler(self, data: torch.Tensor or pd.DataFrame):
        rescale = (data * self.std) + self.mean
        return rescale


def create_process(data: Tuple, preprocessors: List[Preprocess], args: TrainArgs):
    """
    create preprocessors for targets.
    data: (graphs: List, target1, target2, ..)
    preprocessors: a list of scalers.
    args: TrainArgs.
    """
    regression_idx = get_all_index(args.dataset_type, "regression") # The indices of the regression tasks in dataset type list.
    args.check_scaler_num()

    data = list(data)
    for ri, pre in zip(regression_idx, preprocessors):
        if args.scaler_path[ri] is None:
            continue
        print(f"Standardization on {args.task_names[ri]}: mean = {float(pre.mean): .4f}, std = {float(pre.std): .4f}") # 1 means the first element is 'x' (graphs), which is not counted
        data[2 + ri] = pre.scaler(data=data[2 + ri])
    return tuple(data)


def reconstruct_process(preprocessors: List[Preprocess], args: PredictArgs):
    """
    Reconstruct (rescale) the results.
    args: PredictArgs
    """
    regression_idx = get_all_index(args.dataset_type, "regression")
    for ri, pre in zip(regression_idx, preprocessors):
        if pre.path is None:
            continue
        name = args.task_names[ri]

        print(f"Reconstruction on {name}")
        rdf = pd.read_csv(args.save_path[ri])
        rdf[name] = pre.inverse_scaler(rdf)
        rdf.to_csv(args.save_path[ri], index=None)


class MolDataset(Dataset):
    """
    Append main output (y) and other output (multitask)
    """
    def __init__(self, smiles, graphs, target, *args):
        self.smiles = smiles
        self.graphs = graphs
        self.target = target
        self.args = args

    def _get_other_dataset(self, dataset):
        new_dataset = {}
        for i, data in enumerate(dataset):
            new_dataset[i] = data
        return new_dataset

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        if not len(self._get_other_dataset(self.args)):
            return self.smiles[item], self.graphs[item], self.target[item]
        else:
            # if there is only one appended dataset for dataloader
            if not len(self._get_other_dataset(self.args).keys()):
                new_data = self._get_other_dataset(self.args).values()[item]
            else:
                new_data = [values[item] for values in self._get_other_dataset(self.args).values()]
            return self.smiles[item], self.graphs[item], self.target[item], *new_data


def collate(samples):
    dataset = list(map(list, zip(*samples)))
    smiles = dataset.pop(0)
    graphs = dataset.pop(0)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles, bg, *dataset


class MolDataLoader(DataLoader):
    def __init__(self, dataset: MolDataset, batch_size: int = 64, shuffle: bool = False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

        super(MolDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            collate_fn=collate
        )

    def __iter__(self) -> Iterator[MolDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MolDataLoader, self).__iter__()


### This is FFNN dataset and dataloader
class Data(Dataset):
    def __init__(self, X, y, *args):
        self.X = X
        self.y = y
        self.z = args

    def _get_other_dataset(self, dataset):
        new_dataset = {}
        for i, data in enumerate(dataset):
            new_dataset[str(i)] = data
        return new_dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if len(self._get_other_dataset(self.z)) == 0:
            return self.X[item], self.y[item]
        else:
            #if there is only one appended dataset for dataloader
            if len(self._get_other_dataset(self.z).keys()) == 0:
                new_data = self._get_other_dataset(self.z).values()[item]
            else:
                new_data = [values[item] for values in self._get_other_dataset(self.z).values()]

            # results = [self.X[item], self.y[item]]
            # results.extend(new_data)
            return self.X[item], self.y[item], *new_data
