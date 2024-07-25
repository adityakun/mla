#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction using TSNE

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml


# ## Accessing dataset

# In[2]:


X,y=fetch_openml('mnist_784',version=1,return_X_y=True)


# In[3]:


y


# In[4]:


X.shape


# In[5]:


X.iloc[1]


# In[6]:


X.head()


# In[7]:


y.value_counts()


# ## Plotting the images

# In[8]:


plt.imshow(X.iloc[1].to_numpy().reshape(28,28))
plt.title(y[1]);


# In[9]:


plt.imshow(X.iloc[1].to_numpy().reshape(28,28),'Greys')
plt.title(y[1]);


# In[10]:


plt.imshow(X.iloc[10].to_numpy().reshape(28,28),'Greys')
plt.title(y[10]);


# In[11]:


plt.imshow(X.iloc[100].to_numpy().reshape(28,28),'Greys')
plt.title(y[100]);


# In[12]:


plt.imshow(X.iloc[1000].to_numpy().reshape(28,28),'Greys')
plt.title(y[1000]);


# In[13]:


plt.imshow(X.iloc[10000].to_numpy().reshape(28,28),'Greys')
plt.title(y[10000]);


# In[14]:


plt.imshow(X.iloc[60000].to_numpy().reshape(28,28),'Greys')
plt.title(y[60000]);


# ## Creating a subset of data

# In[15]:


np.random.seed(10)
sample=np.random.choice(X.shape[0],1000)
print(sample)


# In[16]:


X1=X.iloc[sample,:]
X1.head()


# In[17]:


y1=y[sample]
y1


# ## Building the TSNE model

# In[19]:


from sklearn.manifold importt TSNE


# In[21]:


tsne=TSNE(n_components=2, perplexity=30,random_state=10)

X_tsne=tsne.fit_transform(X1)
X_tsne.shape


# ## Plotting the transformed points

# In[22]:


plt.scatter(X_tsne[:,0],X_tsne[:,1]);


# In[23]:


plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y1.astype(float));


# ## Creating a DF for better visualisation

# In[24]:


df= pd.DataFrame({'X0':X_tsne[:,0],'X1':X_tsne[:,1],'Label':y1})
df


# In[25]:


df['Label'].value_counts()


# In[26]:


plt.figure(figsize=(15,12))
sns.lmplot(data=df,x='X0',y='X1',hue='Label');


# In[27]:


plt.figure(figsize=(15,12))
sns.lmplot(data=df,x='X0',y='X1',hue='Label',fit_reg=False);


# In[ ]:




