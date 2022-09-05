# Graph-Based-QSPR
A Deep Learning QSPR using graph neural networks.

## Usage:

Run property.py

The following arguments are required.

### Training arguments

1. data path: the file path(s) for training/testing.
```
train_arguments.data_path = ["train_file.csv"]
```
The csv file format:

| smiles                              | homo        | 
| ---------------------------------- | ------------ | 
| Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1	   | -5.383582052 |
| c1ccc2c(c1)CCCO2	                 | -5.325213632 |
| CCCCCCCCCCCCCCC(C)=O	             | -5.490713275 |
| FC(F)(F)/C=C/C(F)(F)F	             | -7.633827539 |
| CSCCC(C)C	                         | -5.002214491 | 

2. save path: the file path for trained model and other generated files.

```
train_arguments.save_path = "save_path"
```

3. model name: the filename of the trained model

```
train_arguments.model_name = "model_name.pkl"
```

4. dataset type: 
I
```
train_arguments.dataset_type = ["regression"]
```

5. loss function:
```
train_arguments.loss_function = ["mse"]
```
7. metric:
8. output dimension:
