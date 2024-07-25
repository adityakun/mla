#!/usr/bin/env python
# coding: utf-8

# # Clustering using Gaussian Mixture Models

# ## Building GMM for a toy dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Creating a dataset

# In[2]:


np.random.seed(100)
X1=np.random.normal(loc=25, scale=6,size=2000)
X1


# In[3]:


sns.distplot(X1,color='red',kde=True);


# In[4]:


X2=np.random.normal(loc=45, scale=5,size=2000)
sns.distplot(X2,color='blue',kde=True);


# In[5]:


X3=np.random.normal(loc=65, scale=4,size=2000)
sns.distplot(X3,color='green',kde=True);


# In[6]:


X4=np.random.normal(loc=85, scale=7,size=2000)
sns.distplot(X4,color='yellow',kde=True);


# In[7]:


# Combaining
X=np.hstack((X1,X2,X3,X4))
X


# ## Finding no of clusters

# In[8]:


from sklearn.mixture import GaussianMixture


# In[9]:


components=np.arange(1,11)
aic_scores=[]
bic_scores=[]
for comp in components:
    model=GaussianMixture(n_components=comp)
    model.fit(X.reshape(-1,1))
    aic=model.aic(X.reshape(-1,1))
    bic=model.bic(X.reshape(-1,1))
    aic_scores.append(aic)
    bic_scores.append(bic)


# In[10]:


# Plotting

plt.plot(components,aic_scores,color='red',label='AIC')
plt.plot(components,bic_scores,color='green',label='BIC')
plt.legend();


# In[11]:


# No of components= clusters=4


# ## Building the model with 4 components

# In[12]:


gmm=GaussianMixture(n_components=4,n_init=10)
gmm.fit(X.reshape(-1,1))


# In[13]:


pred=gmm.predict(X.reshape(-1,1))
pred


# In[14]:


np.unique(pred)


# In[15]:


gmm.means_


# In[16]:


gmm.covariances_


# In[17]:


gmm.weights_


# # Building GMM on nutraline data

# In[18]:


df=pd.read_excel('Nutraline.xlsx',sheet_name='data')
df


# ## Preprocessing

# In[19]:


df.isnull().sum()


# In[22]:


string_columns=df.select_dtypes(include=['object']).columns.tolist()


# In[23]:


string_columns


# In[24]:


# DRoppin

df_1=df.drop(string_columns,axis=1)
df_1


# In[25]:


# Dropping ID
df_1=df_1.drop(['ID'],axis=1)
df_1


# In[26]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(df_1)
X


# In[27]:


df_2=pd.DataFrame(X,columns=df_1.columns)
df_2


# ## Finding no of components

# In[29]:


components=np.arange(1,11)
aic_scores=[]
bic_scores=[]
for comp in components:
    model=GaussianMixture(n_components=comp)
    model.fit(df_2)
    aic=model.aic(df_2)
    bic=model.bic(df_2)
    aic_scores.append(aic)
    bic_scores.append(bic)


# In[30]:


# Plotting

plt.plot(components,aic_scores,color='red',label='AIC')
plt.plot(components,bic_scores,color='green',label='BIC')
plt.legend();


# In[31]:


# No of components=2


# In[32]:


gmm= GaussianMixture(n_components=2,n_init=10)
gmm.fit(df_2)


# In[33]:


pred=gmm.predict(df_2)
pred


# In[34]:


gmm.means_


# In[35]:


gmm.covariances_


# In[36]:


gmm.weights_


# In[37]:


df_1['Cluster_label']=pred
df_1


# In[38]:


df_1['Cluster_label'].value_counts()


# In[39]:


df_1[df_1['Cluster_label']==0]


# In[40]:


df_1[df_1['Cluster_label']==1]


# ## Generative Modelling
# 

# In[41]:


X_new,y_new=gmm.sample(10000)
X_new


# In[ ]:




