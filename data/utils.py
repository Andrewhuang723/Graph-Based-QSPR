import dgl
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from .data import MolDataset, Data
from .feature import Mol2Graph, get_mol
from args import TrainArgs


def read_df(path: List[str] or List[pd.DataFrame]):
    print(f"Tasks: {path} loading...")
    if type(path[0]) == str:
        df_list = [pd.read_csv(f) for f in path]
    else:
        df_list = path
    return df_list


def mask_df(path: List[str] or List[pd.DataFrame], idx: List[int]):
    """
    Delete the unrecognized smiles from original csv.
    """
    for fp in path:
        df = pd.read_csv(fp)
        df.drop(index=idx, inplace=True)
        df.to_csv(fp, index=None)


def get_data(path: List[str] or List[pd.DataFrame], args: TrainArgs, load_graphs_path=None, save_graphs_path=None) -> Tuple[List, List, torch.Tensor]:
    """
    Convert smiles into graphs
    return smiles, graphs, target properties from csv files.
    """
    df_list = read_df(path)
    smiles = df_list[0]["smiles"].tolist()

    if load_graphs_path:
        print(f"Load graphs from {load_graphs_path}.")
        graphs = dgl.data.load_graphs(load_graphs_path)[0]

    else:
        graphs = []
        mask_idx = []
        print("Creating graphs...")
        for i, smi in enumerate(tqdm(smiles, total=len(smiles), leave=False)):
            mol = get_mol(smi, args.is_explicit_H)
            if mol is None:
                mask_idx.append(i)
                print(f"smiles: {smi} is unrecognized.")
                continue
            graph = Mol2Graph(mol)
            graph.is_explicit_H = args.is_explicit_H
            graph.is_shuffle = args.is_shuffle

            graph.addNodes()
            graph.addEdges()

            if args.is_atom_position:
                graph.addAtomPositions()

            graphs.append(graph.Graph)
        print("Creating graphs Done")

        if save_graphs_path:
            print(f"save graphs in {save_graphs_path}")
            dgl.data.save_graphs(save_graphs_path, graphs)

        if len(mask_idx) > 0:
            mask_df(path, mask_idx)

    df_list = read_df(path)
    target_list = [df.drop(columns=["smiles"]).values for df in df_list]
    target = list(map(torch.from_numpy, target_list))

    return smiles, graphs, *target


def get_params(smiles, path):
    """
    Get parameters related to COSMO-SAC model.
    """
    df = pd.read_csv(path)

    params = df.loc[df["smiles"] == smiles][["volume", "sigma_norm"]]
    if len(params) > 1:
        params = params.mean()
    return float(params["volume"]) * 1000, float(params["sigma_norm"])


## For FFNN model
def get_prop_data(path: str, args: TrainArgs):
    xdata = pd.read_csv(path).drop(columns="smiles").values
    df_list = [pd.read_csv(f) for f in args.data_path]
    target_list = [df.drop(columns=["smiles"]).values for df in df_list]
    target = list(map(torch.from_numpy, target_list))

    xdata = torch.from_numpy(xdata).to(device="cuda", dtype=torch.float)
    return xdata, *target


def parse_type(target: List, dataset_type: str):
    if dataset_type == "regression" or dataset_type == "fingerprint":
        if isinstance(target, torch.Tensor):
            return target.to(dtype=torch.float)
        else:
            return torch.stack(target).to(dtype=torch.float)
    else:
        return torch.stack(target).to(dtype=torch.long)


# def split_data(data, test_size: float=0.1) -> Tuple[MolDataset, MolDataset] or Tuple[Data, Data]:
#     """
#     Split data into train and test dataset.
#     Return torch.utils.Dataset or MolDataset
#     """
#     xdata = data[1] # Tuple
#     idx = list(range(len(xdata)))
#     train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42)
#
#     train_ys = []
#     test_ys = []
#     for task in data:
#         if isinstance(task, list):
#             """List: graphs"""
#             select_train_slices = [task[i] for i in train_idx]
#             select_test_slices = [task[i] for i in test_idx]
#         else:
#             """tensor: fingerprints or properties"""
#             select_train_slices = task[train_idx]
#             select_test_slices = task[test_idx]
#         train_ys.append(select_train_slices)
#         test_ys.append(select_test_slices)
#
#     if isinstance(xdata[0], dgl.DGLGraph):
#         return MolDataset(*train_ys), MolDataset(*test_ys)
#     else:
#         return Data(*train_ys), Data(*test_ys)

