from typing import Literal, List
import pandas as pd
from data.feature import PARAMS
import os
import json

class TrainArgs:
    data_path: List[str] or List[pd.DataFrame]
    """x_data is the input file which contains [smiles, x] where x is the input vector rather than smiles"""
    x_data_path: None or str
    """The path for saving model and results and figures"""
    save_path: str
    """The saving model name"""
    model_name: str

    """Task type"""
    dataset_type: List[Literal['regression', 'classification', 'fingerprint', 'smiles']]
    epoch: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    loss_function: List[Literal['mse', 'cross_entropy', 'binary_cross_entropy']]
    metric: List[Literal['mse', 'mae', 'binary_accuracy', 'multiclass_accuracy']]
    early_stopping: int = 50
    """For multiclass -> n_classes"""
    multiclass_num: List[int]
    output_dim: List[int]
    """Storing the mean and std for preprocessing"""
    scaler_path: List[str] = None
    is_explicit_H: bool = False
    """Load pretrained model"""
    pretrained_model: str = None

    """shuffled nodes"""
    is_shuffle: bool = False

    """NN model type"""
    NN: str = "MPNN" # or FFNN or Set2SetNN or ReadoutNN

    """readout type"""
    readout: str = "Set2Set" # or sum or mean

    """atom position add into node features"""
    is_atom_position: bool = False

    """output the hidden states of the model"""
    output_hidden_states: bool = False

    """output the predictions"""
    output_predictions: bool = False

    """save graphs path"""
    save_graphs_path: str = None

    """load graphs path"""
    load_graphs_path: str = None

    """Selective sampling"""
    selective_sampling: dict = None

    # model parameters
    input_dim: int = None # for FFNN model only
    node_hidden_dim: int = 50
    edge_hidden_dim: int = 50
    node_feat_dim: int = PARAMS.ATOM_FDIM
    edge_feat_dim: int = PARAMS.BOND_FDIM
    mpnn_steps: int = 6
    s2s_steps: int = 12
    s2s_layer_nums: int = 3

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._num_tasks = None

    @property
    def task_names(self) -> List[str]:
        """
        Define tags for tasks, useful in multitask learning
        """
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    def check_valid_multiclass(self):
        multiclasses_type_num = self.dataset_type.count("classification")
        if len(self.multiclass_num) != multiclasses_type_num:
            raise ValueError(f"multiclass_num {self.multiclass_num} not equal to the number of classification datasetype {multiclasses_type_num}")

    def check_scaler_num(self):
        scaler_num = self.dataset_type.count("regression")
        if len(self.scaler_path) != scaler_num:
            raise ValueError(f"Length of scaler_path {len(self.scaler_path)} not equal to the number of regression datasetype {scaler_num}")

    def _is_valid_save_path(self):
        if not os.path.exists(self.save_path):
            raise ValueError(f"There is no save path name: {self.save_path}")

    def _is_valid_NN(self):
        if self.NN == "FFNN":
            if not self.input_dim or not self.x_data_path:
                raise ValueError(f"You are running FFNN model but no input_dim.")

    def _save_config(self):
        with open(os.path.join(self.save_path, "config.json"), "w") as fp:
            json.dump(self.__dict__, fp)

    def _is_valid_selective_sampling(self):
        if self.selective_sampling:
            params = ["M", "N", "task"]
            params = set(params)
            ss = set(self.selective_sampling.keys())
            if ss != params:
                raise ValueError("selective sampling parameters (M, N, task) are not defined.")


class PredictArgs:
    data_path: List[str] or List[pd.DataFrame]
    x_test_data_path: str or None
    save_path: List[str]
    model_path: str
    batch_size: int = 128
    dataset_type: List[Literal['regression', 'classification', 'fingerprint', 'smiles']]
    loss_function: List[Literal['mse', 'cross_entropy', 'binary_cross_entropy']]
    metric: List[Literal['mse', 'mae', 'binary_accuracy', 'multiclass_accuracy']]
    output_dim: List[int]
    is_shuffle: bool = False
    output_hidden_states: bool = False
    output_predictions: bool = True
    save_graphs_path: str = None
    load_graphs_path: str = None

    def __init__(self, *args, **kwargs) -> None:
        super(PredictArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._num_tasks = None

    @property
    def task_names(self) -> List[str]:
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0