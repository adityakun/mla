#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


wine=pd.read_csv('wine.csv')
wine


# In[3]:


wine.isnull().sum()


# ## Preprocessing

# In[4]:


y=wine['Wine']
y


# In[5]:


y.unique()


# In[6]:


# Dropping y

X= wine.drop(['Wine'],axis=1)
X


# In[7]:


X.shape


# ## Standardisation

# In[8]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled


# ## Construction of covairance matrix

# In[9]:


cm=np.cov(X_scaled.T)


# In[10]:


cm


# In[11]:


cm.shape


# ## Finding eigen value, eigen vector

# In[12]:


eig_val,eig_vec=np.linalg.eig(cm)
eig_val


# In[13]:


eig_vec


# ## Sorting eigen values

# In[14]:


sorted_eig_val=[i for i in sorted(eig_val, reverse=True)]
sorted_eig_val


# ## Finding cumulative value

# In[15]:


tot=sum(sorted_eig_val)
tot


# In[16]:


exp_var=[(i/tot) for i in sorted_eig_val]
exp_var


# In[17]:


cum_exp_var=np.cumsum(exp_var)
cum_exp_var


# ## Plotting

# In[18]:


plt.bar(range(1,14),exp_var,label='Explained variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained variance')
plt.legend();


# ## Construction of a projection matrix

# In[19]:


eigen_pair=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eigen_pair


# ## Choosing dimension to be 2

# In[20]:


w=np.hstack((eigen_pair[0][1][:,np.newaxis],
            eigen_pair[1][1][:,np.newaxis]))


# In[21]:


w


# In[22]:


w.shape


# ## Ttransforming 13 dim to 2 dim

# In[23]:


X_scaled.shape


# In[24]:


w.shape


# In[25]:


new_X=X_scaled.dot(w)
new_X


# In[26]:


new_X.shape


# ## Visualising 

# In[27]:


for l in np.unique(y):
    plt.scatter(new_X[y==1,0],new_X[y==1,1],marker='s')
    plt.scatter(new_X[y==2,0],new_X[y==2,1],marker='x')
    plt.scatter(new_X[y==3,0],new_X[y==3,1],marker='*')


# ## Using sklearn

# In[28]:


from sklearn.decomposition import PCA


# In[29]:


pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)


# In[30]:


X_pca


# In[31]:


pca.components_.T[:,0]


# In[32]:


pca.components_.T[:,1]


# In[33]:


pca.explained_variance_ratio_


# In[ ]:




