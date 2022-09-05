import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from args import PredictArgs, TrainArgs
from data.utils import get_data, get_prop_data
from data.data import MolDataLoader, MolDataset, Preprocess, create_process, reconstruct_process, Data
from model.model import MPNN, FFNN, Set2SetNN, ReadoutNN
from .test import predict
from .utils import accuracy, multi_accuracy, cross_entropy

device = torch.device("cuda")

valid_loss_function = ['mse', 'cross_entropy', 'binary_cross_entropy']
valid_metric = ['mse', 'mae', 'binary_accuracy', 'multiclass_accuracy']


def run_testing(args: PredictArgs, train_args: TrainArgs):
    if train_args.NN == "MPNN":
        test_data = get_data(args.data_path, train_args, args.load_graphs_path, args.save_graphs_path)
        NN = MPNN

    elif train_args.NN == "Set2SetNN":
        test_data = get_data(args.data_path, train_args, args.load_graphs_path, args.save_graphs_path)
        NN = Set2SetNN

    # Set2Set model
    elif train_args.NN == "ReadoutNN":
        test_data = get_data(args.data_path, train_args, args.load_graphs_path, args.save_graphs_path)
        NN = ReadoutNN

    else:
        test_data = get_prop_data(args.x_test_data_path, train_args)
        NN = FFNN

    # Preprocessing on regression dataset type
    if train_args.scaler_path:
        preprocessors = [Preprocess(path) for path in train_args.scaler_path]
        test_data = create_process(test_data, preprocessors, train_args)

    test_dataset = MolDataset(*test_data)
    test_dataloader = MolDataLoader(test_dataset, args.batch_size)

    model_dict = torch.load(args.model_path)
    model = NN(train_args).to(device)
    model.load_state_dict(model_dict)

    loss_function_list = []
    for loss_function in args.loss_function:
        if loss_function == "binary_cross_entropy":
            loss_function_list.append(nn.BCELoss().to(device))
        elif loss_function == "cross_entropy":
            loss_function_list.append(cross_entropy)
        elif loss_function == "mse":
            loss_function_list.append(nn.MSELoss().to(device))
        else:
            raise ValueError(f"Unrecognized loss function: {loss_function}. Please type in valid loss functions {valid_loss_function}")

    metric_list = []
    for metric in args.metric:
        if metric == "mse":
            metric_list.append(nn.MSELoss().to(device))
        elif metric == "mae":
            metric_list.append(nn.L1Loss().to(device))
        elif metric == "binary_accuracy":
            metric_list.append(accuracy)
        elif metric == "multiclass_accuracy":
            metric_list.append(multi_accuracy)
        else:
            raise ValueError(f"Unrecognized metric: {metric}. Please type in valid metrics {valid_metric}")

    predict(model, test_dataloader, loss_function_list, metric_list, args, saving=args.output_predictions)

    # Rescale
    if train_args.scaler_path:
        reconstruct_process(preprocessors, args)
