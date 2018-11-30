# 5001 individual_project

Please ignore the comments in the code as I wrote it in Chinese. Codes in ".ipynb" file and ".py" file are the same, just for your convenience. 
Run the '.py' directly with the "train.csv" and "test.csv" in the same path. 

## Environment
windows 10
python 3.5

## Package
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt   
from keras import models  
from keras import layers  
import sklearn  
import xgboost as xgb  
import lightgbm as lgb  
from IPython.core.interactiveshell import InteractiveShell   

## First glance of the data
1.skew-distributed-- lead to high mse in the higher time data.  
2.no null features.  
3. small size of sample may lead to overfitting.  

## Data cleaning and feature
1.Delete some features:['random_state','n_informative','l1_ratio].  
    Reason: random_state is useless, n_informative is highly similar to another feature(forgot it), l1_ratio is the same as penalty.  
2.add new features:  
2.1 sample(>600 return 1)  

```python
def sample(num):
    if num>600:
        return 1
    else: 
        return 0
```

2.2 plus

```python
joint2['plus']=joint2['n_features']*joint2['n_samples']*joint2['max_iter']*joint2['n_classes']/joint2['n_jobs']
```

3. n_job=-1 convert to n_job=16  

## Model selection and parameter adjustment

I have tried Keras neuron network, Lasso, KRR, Enet, SVR, XGB, LGB, RandomForest and so on. Neuron network achieve the best score in the publicboard but it got the largest std in local validation, so I finally used the KRR model. And adjust the parameter according to grid search.  
why neuron network and boosting method do not work in my model: I suppose that I have deleted so many features that these ensemble or complex model may lead to overfitting. If I create more features, it may be BETTER.

## Final rank 
top 20, mse 1.02397 in the privateboard.

