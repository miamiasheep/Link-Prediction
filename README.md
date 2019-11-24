# Purdue_ML_Course_Project
ML project in Purdue Fall, 2019

All the experiements are described in the following report.

https://github.com/miamiasheep/Purdue_ML_Final_Project/blob/master/final_report.pdf

## How to run the program

#### Download the dependencies
We suggest you use pyenv or virtualenv before executing following commands.

If you don't want to use pyenv or virtualenv, please at least look at `requirement.txt` and see what packages we will download for you. 

```
pip install -r requirement.txt
```

### Heuristic models

You can reproduce all of our experiments about heuristic model using the following command.

Input is the data set we already collected in the `data/` directory.

```
python main.py --input [file_name1,file_name2,...] --goal [auc/acc]
```

If you use acc option you can use --at option for the k in precision@k

If you don't specify --at argument, we will use the k that equal to the label size of each data set.

ex:
```
python main.py --input [file_name1, file_name2, ...] --goal acc --at 10
```

### Matrix Factorization

#### Disclaimer

libmf is not made by us. It is a tool we download from:

https://www.csie.ntu.edu.tw/~cjlin/libmf/

#### How to reproduce our result

First, you have to compile the libmf by `make` commands in libmf directory.

If you want to use matrix factorization, you can execute:

```
python mf.py --goal [auc/f1] --dup [yes/no]
```

It will perform matrix factorization on all of the data set. 

### Result

All the result will be at `/result` directory.

`result/auc.csv` record the result of auc in different data set using different heuristic methods.

`result/acc_k.csv` record the result of precision@k in different data set using different heuristic methods.

`result/mf_auc.csv` record the result of auc in different data set using matrix factorzation.

`result/mf_dup_auc.csv` record the result of auc in different data set using matrix factorzation using duplicate data.

`result/mf_f1.csv` record the result of f1@(label size) in different data set using matrix factorization.

`result/mf_dup_f1.csv` record the result of f1@(label_size) in different data set using matrix mactorization using duplicate data.

`/result/f1.csv` record the result of f1 score @ (#positive labels) in different data set using different methods.

## ensemble heuristics
You can find training and testing sets with heuristics values as feature vectors in `/ml_heuristics`. 

With those datasets, you can easily build a machine learning model. We use randomforest and logistic regression for examples. You can see the examples in `/ml_heuristics/playground.ipynb`. And the stored results (auc/f1) can be found in `/result/[auc, f1].csv`. 

