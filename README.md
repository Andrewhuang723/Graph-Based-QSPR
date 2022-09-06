# Graph-Based-QSPR
A Deep Learning QSPR using graph neural networks for molecular property predictions.

<p align="center">
<img src="./docs/diagram.png", width="1000"/>
</p>

## Usage:

Run property.py

### Training arguments
**The following training arguments are required.**
- data path: file path(s) of training dataset.
- save path: save path: file path of trained model and other generated files.
- model name: filename of the trained model
- dataset type: task(s) of the target property, including classification, regression, fingerprint and smiles. 
- task name: name(s) of the target property.
- loss function
- metric
- output dimension
```
train_arguments.data_path = ["train_file.csv"]
train_arguments.save_path = "save_path"
train_arguments.model_name = "model_name.pkl"
train_arguments.dataset_type = ["regression"]
train_arguments.task_names = ["homo"]
train_arguments.loss_function = ["mse"]
train_arguments.loss_function = ["mse"]
train_arguments.output_dim = [1]
```
 *csv format of training (or testing) dataset file:

| smiles                             | homo         | 
| ---------------------------------- | ------------ | 
| Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1	   | -5.383582052 |
| c1ccc2c(c1)CCCO2	                 | -5.325213632 |
| CCCCCCCCCCCCCCC(C)=O	             | -5.490713275 |
| FC(F)(F)/C=C/C(F)(F)F	             | -7.633827539 |
| CSCCC(C)C	                         | -5.002214491 | 

### Testing arguments
**The following testing arguments are required.**
- data path: file path(s) of testing dataset.
- save path: file path(s) of prediction results.
- model path: file path of the model for prediction.
- dataset type: task(s) of the target property, including classification, regression, fingerprint and smiles. 
- task name: name(s) of the target property.
- loss function
- metric
- output dimension
```
test_arguments.data_path = ["test_file.csv"]
test_arguments.save_path = ["test_pred.csv"]
test_arguments.model_path = "model.pkl"
test_arguments.task_names = [propName]
test_arguments.dataset_type = ["regression"]
test_arguments.loss_function = ["mse"]
test_arguments.metric = ["mae"]
```
