# Graph-Based-QSPR
A Deep Learning QSPR using graph neural networks for molecular property predictions.

## Usage:

Run property.py

The following arguments are required.

### Training arguments

1. data path: file path(s) of training dataset.
```
train_arguments.data_path = ["train_file.csv"]
```
  The csv file format:

| smiles                             | homo         | 
| ---------------------------------- | ------------ | 
| Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1	   | -5.383582052 |
| c1ccc2c(c1)CCCO2	                 | -5.325213632 |
| CCCCCCCCCCCCCCC(C)=O	             | -5.490713275 |
| FC(F)(F)/C=C/C(F)(F)F	             | -7.633827539 |
| CSCCC(C)C	                         | -5.002214491 | 

2. save path: file path of trained model and other generated files.
```
train_arguments.save_path = "save_path"
```
3. model name: filename of the trained model
```
train_arguments.model_name = "model_name.pkl"
```
4. dataset type: task(s) of the target property, including classification, regression, fingerprint and smiles. 
```
train_arguments.dataset_type = ["regression"]
```
5. task name: name(s) of the target property.
```
train_arguments.task_names = ["homo"]
```
6. loss function:
```
train_arguments.loss_function = ["mse"]
```
7. metric:
```
train_arguments.loss_function = ["mse"]
```
8. output dimension:
```
train_arguments.output_dim = [1]
```

### Testing arguments
1. data path: file path(s) of testing dataset.
```
test_arguments.data_path = ["test_file.csv"]
```
2. save path: file path(s) of prediction results.
```
test_arguments.save_path = ["test_pred.csv"]
```
3. model path: file path of the model for prediction.
```
test_arguments.model_path = "model.pkl"
```
4. dataset type, task name, loss function and metric were same as training arguments:
```
test_arguments.task_names = [propName]
test_arguments.dataset_type = ["regression"]
test_arguments.loss_function = ["mse"]
test_arguments.metric = ["mae"]
```
