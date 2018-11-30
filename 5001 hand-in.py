
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,RANSACRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from IPython.core.interactiveshell import InteractiveShell 
#####just try different kind sof models

InteractiveShell.ast_node_interactivity = "all"
# In[3]:


print('=====load dataset=====')
train_set=pd.read_csv('D:\\18\\bdt\\5001\\personal project\\data\\train.csv',header='infer')
test_set=pd.read_csv('D:\\18\\bdt\\5001\\personal project\\data\\test.csv',header='infer')

train_set.info()
train_set.shape
train_set.describe()
test_set.info()
test_set.head()
test_set.shape


# In[4]:


corrmat = train_set.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True)#heat plot

sns.set()
cols = ['penalty','l1_ratio','alpha','max_iter','random_state','n_jobs','n_samples','n_features','n_classes','n_informative','flip_y','scale']
sns.pairplot(train_set[cols], height = 2.5)
plt.show();

sns.pairplot(test_set[cols], height = 2.5)
plt.show();


# Checking for outliers in GrLivArea as indicated in dataset documentation
sns.regplot(x=train_set['random_state'], y=train_set['time'], fit_reg=True)
plt.show()

# Checking for outliers in GrLivArea as indicated in dataset documentation
sns.regplot(x=train_set['max_iter'], y=train_set['time'], fit_reg=True)
plt.show()

# Checking for outliers in GrLivArea as indicated in dataset documentation
sns.regplot(x=train_set['n_samples'], y=train_set['time'], fit_reg=True)
plt.show()

# Checking for outliers in GrLivArea as indicated in dataset documentation
sns.regplot(x=train_set['n_jobs'], y=train_set['time'], fit_reg=True)
plt.show()


sns.regplot(x=train_set['n_clusters_per_class'], y=train_set['time'], fit_reg=True)
plt.show()

train_set.shape


# # Feature engineering

# In[5]:


y=train_set['time'].copy()
logy=np.log1p(y)
train_set=train_set.drop(columns=['time'])
train_set=train_set.drop(columns=['id'])
#train_set=train_set.drop(columns=['n_jobs'])
#train_set=train_set.drop(columns=['n_clusters_per_class'])
ID=test_set['id']
test_set=test_set.drop(columns=['id'])
#test_set=test_set.drop(columns=['n_jobs'])
#test_set=test_set.drop(columns=['n_clusters_per_class'])
test_set.shape
train_set.shape
train_set.isnull().sum()  #### 
test_set.isnull().sum() ####similarly


# # feature engineering 2

# In[6]:


####  n_job   -1  变16
####random state  删掉
###n_informative  删掉
train_set.n_jobs[train_set.n_jobs ==-1] = 16
test_set.n_jobs[test_set.n_jobs ==-1] = 16
train_set=train_set.drop(columns=['random_state'])
test_set=test_set.drop(columns=['random_state'])

train_set=train_set.drop(columns=['n_informative'])
test_set=test_set.drop(columns=['n_informative'])
train_set=train_set.drop(columns=['l1_ratio'])
test_set=test_set.drop(columns=['l1_ratio'])


# # distribution change and new features

# In[8]:


def sample(num):
    if num>600:
        return 1
    else: 
        return 0


# In[9]:



joint2 = pd.concat([train_set, test_set], axis=0)
joint2['plus']=joint2['n_features']*joint2['n_samples']*joint2['max_iter']*joint2['n_classes']/joint2['n_jobs']
joint2['smaple']=joint2['n_samples'].apply(sample)
# Fix for skewness


# In[10]:


#data['feature_1'] = data['max_iter'] * data['n_samples'] * data['n_features'] * data['n_classes'] / data['n_jobs']

joint2['scale']=np.log1p(joint2['scale'])
train_set = joint2.head(train_set.shape[0])
test_set= joint2.tail(test_set.shape[0])


# In[11]:


cat2=[ 'alpha', 'max_iter', 'n_jobs', 'n_samples', 'n_features',
       'n_classes', 'n_clusters_per_class', 'flip_y', 'scale', 'plus',
       'smaple']
for feature in cat2:
    train_set[feature] = (train_set[feature]-test_set[feature].min())/test_set[feature].std()
    test_set[feature] = (test_set[feature]-test_set[feature].min())/test_set[feature].std()


# In[12]:


test_set.describe()


# # dummy

# In[13]:


joint = pd.concat([train_set, test_set], axis=0)
cat= joint.select_dtypes(include=['object']).axes[1]
for col in cat:
    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)
    joint.drop(col, axis=1, inplace=True)
train_dum = joint.head(train_set.shape[0])
test_dum= joint.tail(test_set.shape[0])


# In[14]:


train_dum.shape
test_dum.shape
y.describe()


# # 数据标准

# In[15]:


#from sklearn.decomposition import PCA
'''
pca=PCA(n_components=10)
train_dum=pca.fit_transform(train_dum)
test_dum=pca.transform(test_dum)
'''
####standardize the numerical features
scaler = StandardScaler()
scaler.fit(train_dum)
train_dum= scaler.transform(train_dum)
test_dum= scaler.transform(test_dum)


# # 模型选择     NN

# In[16]:



def build_model():
    '''
    
    '''
    model = models.Sequential()
    model.add(layers.Dense(25, activation='relu', input_shape=(train_dum.shape[1],)))
   # model.add(layers.Dense(500,activation='relu'))
    #model.add(layers.Dense(200,activation='relu'))
   # model.add(layers.Dense(100,activation='relu'))
   # model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(25,activation='relu'))
    model.add(layers.Dense(1,activation='relu'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# ### cross validation of nn

# In[17]:


import numpy as np
k = 4
num_val_samples = len(train_dum) // k #整数除法
num_epochs =15
all_scores = []
for i in range(k):
    print('processing fold #', i)
    #依次把k分数据中的每一份作为校验数据集
    val_data = train_dum[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = y[i* num_val_samples : (i+1) * num_val_samples]
    
    #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
    partial_train_data = np.concatenate([train_dum[: i * num_val_samples], 
                                         train_dum[(i+1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([y[: i * num_val_samples], 
                                            y[(i+1) * num_val_samples: ]],
                                          axis = 0)
    print("build model")
    model = build_model()
    #把分割好的训练数据和校验数据输入网络
    model.fit(partial_train_data, partial_train_targets, epochs =30, 
              batch_size =16, verbose = 0)
    print("evaluate the model")
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mse)
    
print(all_scores)


# In[18]:


model = build_model()
model.fit(train_dum, y, epochs = 40, batch_size =20, verbose = 0)
p1=model.predict(test_dum)
p2=model.predict(train_dum)


# In[20]:



plt.hist(p1)

plt.show()


# In[19]:


from sklearn.metrics import mean_squared_error
mse2=mean_squared_error(y, p2)
mse2


# # 建模   
# 

# In[21]:


'''submission=pd.DataFrame()
submission['id']=ID
submission['time']=p1
submission.to_csv('keras1119f3.csv',index=False)'''


# # KRR  the best

# In[23]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=True).get_n_splits(train_dum)
    rmse= np.sqrt(-cross_val_score(model, train_dum, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[24]:



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.5, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.5, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.7)

GBoost = GradientBoostingRegressor()

model_xgb = xgb.XGBRegressor()

model_lgb = lgb.LGBMRegressor(objective='regression')
svr_rbf = SVR(kernel='rbf',C=1e1, gamma='auto')
svr_lin = SVR(kernel='linear',gamma='scale')
svr_poly = SVR(kernel='poly',degree=2,gamma='scale')

bridge=BayesianRidge()
icll=LassoLarsIC()

rf=RandomForestRegressor(n_estimators=3)

ra=RANSACRegressor()


# In[25]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("\nKRR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("\nmodel_xgbt score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("\nmodel_lgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score_rbf = rmsle_cv(svr_rbf)
print("\nrbf score: {:.4f} ({:.4f})\n".format(score_rbf.mean(), score_rbf.std()))
score_lin=rmsle_cv(svr_lin)
print("\nlin score: {:.4f} ({:.4f})\n".format(score_lin.mean(), score_lin.std()))
score_poly = rmsle_cv(svr_poly)
print("\npoly score: {:.4f} ({:.4f})\n".format(score_poly.mean(), score_poly.std()))

score_br=rmsle_cv(bridge)
print("\bridge score: {:.4f} ({:.4f})\n".format(score_br.mean(), score_lin.std()))

score_icll = rmsle_cv(icll)
print("\icll score: {:.4f} ({:.4f})\n".format(score_icll.mean(), score_poly.std()))

score_rf= rmsle_cv(rf)
print("\ rfscore: {:.4f} ({:.4f})\n".format(score_rf.mean(), score_poly.std()))


# ## KRR performs the best

# In[26]:



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.7)
model1=KRR.fit(train_dum,y)
predictions =KRR.predict(train_dum)
###accuracy of training set
explogp22=np.expm1(predictions)

explogp22[explogp22<0.07]=0.07
mse =mean_squared_error(y, predictions)
print(mse)



p1=KRR.predict(test_dum)

explogy=np.expm1(p1)

explogy[explogy<0.07]=0.07
p1[p1<0.07]=0.07
submission=pd.DataFrame()
submission['id']=ID
submission['time']=p1
submission.to_csv('keridge1119444.csv',index=False)

