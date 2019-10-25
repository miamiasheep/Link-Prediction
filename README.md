# Purdue_ML_Course_Project
ML project in Purdue Fall, 2019

## How to run the program
I suggest you use pyenv or virtualenv before executing following commands.

If you don't want to use pyenv or virtualenv, please at least look at `requirement.txt` and see what packages we will download for you. 

```
pip install -r requirement.txt
python main.py --input [file_name1,file_name2,...]
```

All the result will be at `/result` directory.

`/result/auc.csv` record the result of auc in different data set using different methods.

`/result/f1.csv` record the result of f1 score @ (#positive labels) in different data set using different methods.
