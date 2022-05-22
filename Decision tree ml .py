#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from sklearn.decomposition import PCA, KernelPCA

from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('forestfires.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


df.describe()


# In[9]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidths=0.5)


# # DATA CLEANING

# In[10]:


df['rain'].value_counts()


# In[11]:



df.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)


# In[12]:


df.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'),(1,2,3,4,5,6,7), inplace=True)


# In[13]:


print(df.head())


# # box and whisket plot

# In[14]:


df.plot(kind='box', subplots=False, sharex=False, sharey=False, layout=(4,4),figsize=(10, 10))
plt.suptitle("Outliers", y=1.00, fontweight='bold', fontsize=40)
plt.show()


# # pair plot

# In[15]:


sns.pairplot(df)
plt.show()


# # How are the forest fires distributed in the park

# In[16]:


sns.distplot(df["area"])


# In[17]:


print(df["area"].describe().T)
plt.boxplot(df["area"])


# In[18]:


print(len(df[df["area"]==0.0])*100/df.shape[0])


# # Highest, Lowest and average affected area value :

# In[19]:


print(df["area"].agg(["max","min","mean"]))


# # How many times this avg area or more than avg got affected 

# In[20]:


len(df[df["area"]>=12.84])*100/df.shape[0]


# How many times in % the maximum area got affected ?

# In[21]:


len(df[df["area"]==1090.84])*100/df.shape[0]


# What was position of this highly affected area :

# In[22]:


df[df["area"]== 1090.840000][["X","Y"]]


# What was the temp,rain,wind,RH and Moisture codes(FFMC,DMC,DC) values corresponding to this highly burnt area

# In[23]:


x=df[df["area"]==1090.84]
x.iloc[:,2:12]


# Which month is more likely to forest fire

# In[24]:


print(df["month"].value_counts())


# In[25]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.hist(df["month"],bins=8)
plt.xlabel(" months ")
plt.ylabel("frequency")


# In[26]:


X, y = df.drop('area', 1).values, df['area'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[27]:


scaler = StandardScaler()
scaler.fit(X_train)


# In[28]:


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


# # Models

# In[29]:


def model_results(model, X_train, y_train):
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.3f}\n")
    
    print("*****Shapiro-Wilks test for Normality*****")
    residuals = y_train - predictions
    test_results = stats.shapiro(residuals)
    if test_results[1]>0.5:
        print("\nResiduals follow normal distribution!")
    else:
        print("\nResiduals do no follow normal distribution!")
    
    plt.scatter(np.arange(len(residuals)), residuals)
    plt.title("Residual Plot")
    plt.show()


# Linear Regression model

# In[30]:


linear_reg = LinearRegression()

model_results(linear_reg, X_train_scaled, y_train_log)


# lasso

# In[31]:


lasso = Lasso(alpha=0.2, random_state=101)

model_results(lasso, X_train_scaled, y_train_log)


# Ridge

# In[32]:


ridge = Ridge(alpha=0.1, random_state=101)

model_results(ridge, X_train_scaled, y_train_log)


# Elastic Net

# In[34]:


elastic_net = ElasticNet(alpha=0.1)
model_results(elastic_net, X_train_scaled, y_train_log)


# In[35]:


elastic_net.coef_


# In[36]:


## get important features only

indexes = np.where(elastic_net.coef_!=0)[0].tolist()
indexes


# In[37]:


elastic_net_2 = ElasticNet(alpha=0.1)
model_results(elastic_net_2, X_train_scaled[:, indexes], y_train_log)


# In[38]:


def train_test_error(model, X_train=X_train_scaled, 
                     X_test=X_test_scaled, y_train=y_train_log, 
                     y_test=y_test_log):
    
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}")


# SVR

# In[39]:



svr_reg = SVR(kernel='rbf', degree=2, C=1)
svr_reg.fit(X_train_scaled, y_train_log)

train_test_error(svr_reg)


# Random Forest

# In[40]:


forest = RandomForestRegressor(n_estimators=10, random_state=101)
forest.fit(X_train_scaled, y_train_log)

train_test_error(forest)

