from train.run_train import run_training
from train.run_test import run_testing
from args import TrainArgs, PredictArgs
import pandas as pd
from tools import *

propName = "homo"
model_path = f"tests/{propName}_MODEL"

train_arguments = TrainArgs()
train_arguments.data_path = [f"tests/data/{propName}_train.csv"]
train_arguments.save_path = model_path
train_arguments.load_graphs_path = f"tests/data/{propName}_train_graphs.bin" # execute if graphs were saved previously.
train_arguments.model_name = f"{propName}.pkl"
train_arguments.epoch = 1000
train_arguments.lr = 5e-4
train_arguments.batch_size = 128
train_arguments.early_stopping = 50
train_arguments.is_explicit_H = True
train_arguments.dataset_type = ["regression"]
train_arguments.task_names = [propName]
train_arguments.scaler_path = [f"tests/data/{propName}_scaler.csv"] # mean & std
train_arguments.output_dim = [1]
train_arguments.loss_function = ["mse"]
train_arguments.metric = ["mae"]
train_arguments.readout = "sum"
train_arguments.selective_sampling = {"M": 0.2, "N": 5, "task": propName}

test_arguments = PredictArgs()
test_arguments.data_path = [f"tests/data/{propName}_test.csv"]
# test_arguments.load_graphs_path = f"tests/data/{propName}_test_graphs.bin" # execute if graphs were saved previously.
test_arguments.save_path = [f"{model_path}/{propName}_test_pred.csv"]
test_arguments.model_path = f"{model_path}/{propName}.pkl"
test_arguments.task_names = [propName]
test_arguments.dataset_type = ["regression"]
test_arguments.loss_function = ["mse"]
test_arguments.metric = ["mae"]

if __name__ == "__main__":

    # training
    train_arguments._is_valid_save_path()
    train_arguments._save_config()

    model = run_training(train_arguments)

    # predict
    results = run_testing(test_arguments, train_arguments)

    # testing & prediction
    test_df = pd.read_csv(f"tests/data/{propName}_test.csv")
    pred_df = pd.read_csv(f"{model_path}/{propName}_test_pred.csv")

    R2_plot(pred_df[propName].values, test_df[propName].values, propName, save_dir=model_path)
