# Graph-Based-QSPR
A Deep Learning QSPR using graph neural networks.

## Usage:

Run property.py

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

