# Purdue_ML_Course_Project
ML project in Purdue Fall, 2019

## How to run the program
```
pip install -r requirement.txt
python main.py --input [file_name]
```

Kernel是自己定的，所以我們才要學Mercer theorem。
雖然我不知道我定的kernel是對應到哪個phi function，
但我知道只要滿足Mercer theorem的條件，
我的kernel function是勢必可以找到對應的phi function。

然後知道Kernel後我就可以避免掉原本 phi(x)* phi(y) 的運算。
因為我們用K(x,y)去換掉了原本的phi(x) * phi(y)，
然後這其實會加速非常多，
有個很好的例子是上課所教的無限多維度的例子。
