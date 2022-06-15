#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

import scipy 

import statsmodels.api as sm

import sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing
from sklearn.decomposition import PCA


# In[4]:


df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Covered Hoppers Overall.csv",index_col='Date',parse_dates=True)


# # Visulalizations

# In[5]:


import numpy as np

df['Time']= np.arange(len(df.index))


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

plt.rc("figure",autolayout=True)

plt.rc("axes")

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

fig, ax=plt.subplots()
ax.plot('Time','Covered_Hopper_lease_rate', data=df,color='0.75')
ax = sns.regplot(x='Time', y='Covered_Hopper_lease_rate',data=df,ci=None,scatter_kws=dict(color='0.25') )
ax.set_title('Time Plot of Lease rates')


# In[7]:


#Yearly Trend
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Covered_Hopper_lease_rate', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.Covered_Hopper_lease_rate, title='Quarterly Covered Hopper Lease Rates')

#Lease Rates were Highest In 2016 


# In[8]:


#Subplots

fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Covered Hoppers Overall.csv", parse_dates=['Date'], index_col='Date').plot(title='OverAll Trend ', legend=True, ax=axes[0])

pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Correlations.csv", parse_dates=['Date'], index_col='Date').plot(title='Carloads Trends ', legend=True, ax=axes[1])

pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Covered Hopper Market analysis.csv", parse_dates=['Date'], index_col='Date').plot(title='Market Trend ', legend=True, ax=axes[2])


# In[9]:


import statsmodels.api as sm

df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Covered Hopper with end market products.csv",index_col='Date',parse_dates=True)


# In[10]:


training = df[0:63]

validation = df[63:86]

forecast = df[75:86]


# In[11]:


y_train=training['Covered_Hopper_lease_rate']
x_train=training.drop(columns = ['Covered_Hopper_lease_rate'])

y_valid=validation['Covered_Hopper_lease_rate']
x_valid=validation.drop(columns = ['Covered_Hopper_lease_rate'])

x_forecast=forecast.drop(columns = ['Covered_Hopper_lease_rate'])


# In[12]:


import sklearn
from sklearn.linear_model import LinearRegression


model= LinearRegression()
model.fit(x_train,y_train)


# In[13]:


X2=sm.add_constant(x_train) # Adding the constant just to get the value of Intercept Coefficient


# In[14]:


est=sm.OLS(y_train,X2)
est2=est.fit()

print(est2.summary())


# In[15]:


y_pred=model.predict(x_valid)


# In[16]:


df=pd.DataFrame({'Actual': y_valid, 'Predicted':y_pred, 'Differrence':y_pred-y_valid})

df.sort_index()


# In[17]:


MAE=sklearn.metrics.mean_absolute_error(y_valid, y_pred)

RMSE=np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_pred))


print("MAE: {0:.3f} ".format(MAE) )

print("RMSE: {0:.3f} ".format(RMSE))


# # Correlation Check

# In[18]:


df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Covered Hopper with end market products.csv",)

#Droping the Date column because already Divided the Dataset using the Date Column so its redundant
df=df.drop(columns=['Date'])


# In[19]:


plt.figure(figsize = (17,10))
matrix = df.corr().round(1)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag',linewidths=.5)
plt.show()


# In[20]:


df1=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Correlations.csv",)


# In[21]:


#Removed the Colums like Total Carloads which offcourse had correlations with other carloads

plt.figure(figsize = (17,10))
matrix = df1.corr().round(1)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag',linewidths=.5)
plt.show()


# # REGRESSION MODEL 2

# In[22]:


df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Correlations.csv",)

df=df.drop(columns=['Date'])


# In[23]:


training = df[0:63]

validation = df[63:86]

forecast = df[75:86]


# In[24]:


y_train=training['Covered_Hopper_lease_rate']
x_train=training.drop(columns = ['Covered_Hopper_lease_rate'])

y_valid=validation['Covered_Hopper_lease_rate']
x_valid=validation.drop(columns = ['Covered_Hopper_lease_rate'])


x_forecast=forecast.drop(columns = ['Covered_Hopper_lease_rate'])


# In[25]:


import sklearn
from sklearn.linear_model import LinearRegression


model= LinearRegression()
model.fit(x_train,y_train)


# In[26]:


X2=sm.add_constant(x_train)


# In[27]:


est=sm.OLS(y_train,X2)
est2=est.fit()

print(est2.summary())


# In[28]:


y_pred=model.predict(x_valid)


# In[29]:


df=pd.DataFrame({'Actual': y_valid, 'Predicted':y_pred,'Differrence':y_pred-y_valid})

df.sort_index()


# In[30]:


MAE=sklearn.metrics.mean_absolute_error(y_valid, y_pred)

RMSE=np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_pred))


print("MAE: {0:.3f} ".format(MAE) )

print("RMSE: {0:.3f} ".format(RMSE))


# # REGRESSION MODEL 3 with PCA

# Finding the threshold for the number of componenets to be used for PCA

# In[31]:


df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Correlations.csv",)

df=df.drop(columns=['Date'])


# In[36]:


pca = PCA().fit(df)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 16, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) 
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.99, color='r', linestyle='-')
plt.text(0.5, 0.85, '99% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# In[37]:


training = df[0:63]

validation = df[63:86]


forecast = df[75:86]


# In[38]:


y_train=training['Covered_Hopper_lease_rate']
x_train=training.drop(columns = ['Covered_Hopper_lease_rate'])

y_valid=validation['Covered_Hopper_lease_rate']
x_valid=validation.drop(columns = ['Covered_Hopper_lease_rate'])

x_forecast=forecast.drop(columns = ['Covered_Hopper_lease_rate'])


# In[39]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x_train)
pca.fit(x_valid)
x_train = pca.transform(x_train)
x_valid=pca.transform(x_valid)

print("transformed shape:", x_train.shape)
print("transformed shape:", x_valid.shape)


# In[40]:


import sklearn
from sklearn.linear_model import LinearRegression


model= LinearRegression()
model.fit(x_train,y_train)


# In[41]:


X2=sm.add_constant(x_train)


# In[42]:


est=sm.OLS(y_train,X2)
est2=est.fit()

print(est2.summary())


# In[43]:


y_pred=model.predict(x_valid)


# In[44]:


df=pd.DataFrame({'Actual': y_valid, 'Predicted':y_pred,'Differrence':y_pred-y_valid})

df.sort_index()


# In[45]:


MAE=sklearn.metrics.mean_absolute_error(y_valid, y_pred)

RMSE=np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_pred))

print("MAE: {0:.3f} ".format(MAE) )

print("RMSE: {0:.3f} ".format(RMSE))


# In[ ]:





# In[ ]:





# In[ ]:




