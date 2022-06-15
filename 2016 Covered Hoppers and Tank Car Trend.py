#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[22]:


df=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Year2016.csv",index_col='Date',parse_dates=True)


# In[23]:




df['Time']= np.arange(len(df.index))


# In[24]:


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


# In[25]:


def plot_df(df, x, y, title="", xlabel='Date', ylabel='Covered_Hopper_lease_rate', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.Covered_Hopper_lease_rate, title='Quarterly Covered Hopper Lease Rates')

 


# In[26]:


df1=pd.read_csv(r"C:\Users\mahima neema\Desktop\ARLINGTON\PROJECTS\QS Problem Set\QS Problem Set\Year2016 TankCar.csv",index_col='Date',parse_dates=True)


# In[27]:




df1['Time']= np.arange(len(df.index))


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

plt.rc("figure",autolayout=True)

plt.rc("axes")

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

fig, ax=plt.subplots()
ax.plot('Time','Tankcar_lease_rate', data=df1,color='0.75')
ax = sns.regplot(x='Time', y='Tankcar_lease_rate',data=df1,ci=None,scatter_kws=dict(color='0.25') )
ax.set_title('Time Plot of Lease rates')


# In[33]:


def plot_df(df1, x, y, title="", xlabel='Date', ylabel='Tankcar_lease_rate', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()


plot_df(df1, x=df1.index, y=df1.Tankcar_lease_rate, title='Quarterly Tankcar Lease Rates')    


# In[ ]:




