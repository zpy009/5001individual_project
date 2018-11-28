# 5001individual_project

Please ignore the comments in the code as I wrote it in Chinese. Code in .ipynb and .py are the same, just for your convenience. 

## first glance of the data
1.skew-distributed-- lead to high mse in the higher time data.  
2.no null features.  
3. small size of sample may lead to overfitting.  
## data cleaning and feature
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

## model selection and parameter adjustment

I have tried neuron network, Lasso, KRR, Enet, SVR, XGB, LGB, randomforest and so on. Neuron network achieve the best score in the publicboard but it got the largest std in local validation, so I finally used KRR model. And adjust the parameter according to grid search.  
why neuron network and boosting method do not work in my model: I suppose that I have deleted so many features that these ensemble or complex model may lead to overfitting. If I create more features may be BETTER.

## Final rank 
top 20

