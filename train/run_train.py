import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
import time
import os
import pprint
from args import TrainArgs
from data.utils import get_data, get_prop_data
from data.data import MolDataset, MolDataLoader, Preprocess, create_process
from model.model import MPNN, FFNN, Set2SetNN, ReadoutNN
from .train import train
from .test import predict
from .utils import build_optimizer, early_stopping, history_plot, cross_entropy, accuracy, multi_accuracy, SelectiveSample

device = torch.device("cuda")

valid_loss_function = ['mse', 'cross_entropy', 'binary_cross_entropy']
valid_metric = ['mse', 'mae', 'binary_accuracy', 'multiclass_accuracy']


def run_training(args: TrainArgs):
    """
    Run train function.
    """
    args._is_valid_NN()
    args._is_valid_selective_sampling()

    # Message passing model
    if args.NN == "MPNN":
        data = get_data(args.data_path, args, args.load_graphs_path, args.save_graphs_path)  # Tuple(graphs, targets, ...)
        NN = MPNN

    # Set2Set model
    elif args.NN == "Set2SetNN":
        data = get_data(args.data_path, args, args.load_graphs_path, args.save_graphs_path)  # Tuple(graphs, targets, ...)
        NN = Set2SetNN

    # Readout model
    elif args.NN == "ReadoutNN":
        data = get_data(args.data_path, args, args.load_graphs_path, args.save_graphs_path)  # Tuple(graphs, targets, ...)
        NN = ReadoutNN

    # Feed forward network
    else:
        data = get_prop_data(args.x_data_path, args)
        NN = FFNN

    # Preprocessing on regression dataset type
    if args.scaler_path:
        preprocessors = [Preprocess(path) for path in args.scaler_path]
        data = create_process(data, preprocessors, args)

    if args.pretrained_model:
        model_dict = torch.load(args.pretrained_model)
        model = NN(args).to(device)
        model.load_state_dict(model_dict)

    else:
        model = NN(args).to(device)

    loss_function_list = []
    metric_list = []
    for loss_function in args.loss_function:
        if loss_function == "binary_cross_entropy":
            loss_function_list.append(nn.BCELoss().to(device))
        elif loss_function == "cross_entropy":
            loss_function_list.append(cross_entropy)
        elif loss_function == "mse":
            loss_function_list.append(nn.MSELoss().to(device))
        else:
            raise ValueError(f"Unrecognized loss function: {loss_function}. Please type in valid loss functions {valid_loss_function}")

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
            raise ValueError(f"Unrecognized loss function: {metric}. Please type in valid loss functions {valid_metric}")

    if args.selective_sampling:
        M = args.selective_sampling["M"] # M% of samples in training set were selected
        N = args.selective_sampling["N"] # rounds
        task = args.selective_sampling["task"]

        dataset = MolDataset(*data)
        D = len(dataset)
        M_nums = int(D * M)

        train_dataset, test_dataset = random_split(dataset, [M_nums, D - M_nums])

        for i in range(N):
            T = len(train_dataset) # T = M_nums at round 0
            print(f"Round {i} -- TRAINING DATA: {T}; TESTING DATA: {len(test_dataset)}")
            train_dataset_T, val_dataset = random_split(train_dataset, [int(0.9 * T), T - int(0.9 * T)])

            train_dataloader = MolDataLoader(train_dataset_T, args.batch_size)
            val_dataloader = MolDataLoader(val_dataset, args.batch_size)
            test_dataloader = MolDataLoader(test_dataset, args.batch_size)

            optimizer = build_optimizer(model, args.lr)
            best_model_dict, history = epoch_training(model, train_dataloader, val_dataloader, loss_function_list,
                                                      metric_list, optimizer, args)
            model.load_state_dict(best_model_dict)
            collections = predict(model, test_dataloader, loss_function_list, metric_list, args,
                                       saving=False, return_logs=True)

            pred = collections[f"{task}_pred"]
            target = collections[f"{task}_target"]
            loss = torch.abs(pred - target).squeeze(dim=-1)

            # TODO: argmin losses for n samples, and add them to train_dataloader.
            if M_nums > loss.shape[0]: # Cannot sample M_nums of data if testing size of the current round is less than M_nums.
                break
            idx = loss.topk(M_nums)[-1].tolist() # the index of k samples in testing data which are the top k largest errors

            train_dataset, test_dataset = SelectiveSample(train_dataset, test_dataset, idx)

            if len(test_dataset) == 0: # all of the samples in training set is visited.
                break

    else:
        dataset = MolDataset(*data)
        D = len(dataset)
        T = int(D * 0.9) # nums of training dataset

        train_dataset, val_dataset = random_split(dataset, [T, D-T])
        train_dataloader = MolDataLoader(train_dataset, args.batch_size)
        val_dataloader = MolDataLoader(val_dataset, args.batch_size)

        optimizer = build_optimizer(model, args.lr)
        best_model_dict, history = epoch_training(model, train_dataloader, val_dataloader, loss_function_list,
                                                  metric_list, optimizer, args)

    torch.save(best_model_dict, os.path.join(args.save_path, args.model_name))
    print(f"Model {args.model_name} is saved\n")

    history_plot(history, os.path.join(args.save_path, "learning_curve.png"), args.task_names)

    return model


def epoch_training(model, train_dataloader, val_dataloader, loss_function_list, metric_list, optimizer, args):
    epochs = args.epoch
    earlystop = early_stopping(args.early_stopping)

    history = {
        "LOSS": [],
        "VAL_LOSS": [],
        "METRIC": [],
        "VAL_METRIC": []
    }
    # Epoch losses for each task
    if args.num_tasks:
        for task in args.task_names:
            history[f"{task}_LOSS"] = []
            history[f"{task}_VAL_LOSS"] = []
            history[f"{task}_METRIC"] = []
            history[f"{task}_VAL_METRIC"] = []

    checkpoint = {
        "Model": None,
        "Epoch": 0
    }

    for epoch in range(epochs):
        t0 = time.time()

        loss, m = train(model, train_dataloader, loss_function_list, metric_list, optimizer, args)
        val_loss, val_m = predict(model, val_dataloader, loss_function_list, metric_list, args, saving=False)

        loss_sum = sum(loss)
        val_loss_sum = sum(val_loss)

        m_sum = sum(m)
        val_m_sum = sum(val_m)
        history["LOSS"].append(loss_sum)
        history["VAL_LOSS"].append(val_loss_sum)
        history["METRIC"].append(m_sum)
        history["VAL_METRIC"].append(val_m_sum)
        print(f'\nEpoch {epoch:d}')
        print(f'LOSS {loss_sum:.4f} | VAL_LOSS {val_loss_sum:.4f} | METRIC {m_sum:.4f} | VAL_METRIC {val_m_sum:.4f}')

        if args.num_tasks:
            for i, task in enumerate(args.task_names):
                history[f"{task}_LOSS"].append(loss[i])
                history[f"{task}_VAL_LOSS"].append(val_loss[i])
                history[f"{task}_METRIC"].append(m[i])
                history[f"{task}_VAL_METRIC"].append(val_m[i])
                pprint.pprint(
                    f'{task} | LOSS {loss[i]:.4f} | {args.metric[i]} {m[i]:.4f} | VAL_LOSS {val_loss[i]:.4f} | VAL_{args.metric[i]} {val_m[i]:.4f}')
        t1 = time.time()
        print(f"Times {(t1 - t0):.4f}")

        earlystop(val_loss_sum, model)
        if earlystop.early_stop:
            print("Early stopping")
            break

        model_dict = model.state_dict()
        checkpoint[epoch] = model_dict

    min_loss = min(history["VAL_LOSS"])
    min_epoch = history["VAL_LOSS"].index(min_loss)
    print(f"The model in epoch {min_epoch:d} has the minimum loss.\n")

    return checkpoint[min_epoch], history
