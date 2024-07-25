#!/usr/bin/env python
# coding: utf-8

# # Session 1: K Means Clustering

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ## Accessing data
# 

# In[ ]:


X,y=make_blobs(n_samples=1000,n_features=2,centers=1,random_state=10)
plt.figure(figsize=(12,9))

plt.scatter(X[:,0],X[:,1]);


# In[ ]:


X,y=make_blobs(n_samples=1000,n_features=2,centers=2,random_state=10)
plt.figure(figsize=(12,9))

plt.scatter(X[:,0],X[:,1]);


# In[ ]:


X,y=make_blobs(n_samples=1000,n_features=2,centers=4,random_state=10)
plt.figure(figsize=(12,9))

plt.scatter(X[:,0],X[:,1]);


# ## Finding the value of k

# ### 1) Using Elbow method

# In[ ]:


from sklearn.cluster import KMeans

SSD=[]
for k in range(1,25):
    k_means=KMeans(n_clusters=k,random_state=10)
    k_means.fit(X)
    SSD.append(k_means.inertia_)
plt.plot(range(1,25),SSD);


# In[ ]:


# The value of k=4


# ### 2) Using silhouette method

# In[ ]:


from sklearn.metrics import silhouette_score
SS=[]
for k in range(2,25):
    k_means=KMeans(n_clusters=k,random_state=10)
    k_means.fit(X)
    SS.append(silhouette_score(X,k_means.predict(X)))
plt.plot(range(2,25),SS);


# In[ ]:


# The best value of k=4


# ## Building the model

# In[ ]:


best_kmeans=KMeans(n_clusters=4,random_state=10)
best_kmeans.fit(X)


# ## Finding Clusters

# In[ ]:


clusters=best_kmeans.predict(X)
clusters


# In[ ]:


pd.DataFrame(clusters).value_counts()


# ## Cluster centers

# In[ ]:


best_kmeans.cluster_centers_


# ## Plotting clusters and centers

# In[ ]:


plt.figure(figsize=(12,9))
plt.scatter(X[:,0],X[:,1])
plt.scatter(2.57427374,  4.9551547, color='pink')
plt.scatter(5.54690135, -9.62123904, color='violet')
plt.scatter(-6.10307996,  5.14422118, color='red')
plt.scatter(-0.03749354, -5.43011018, color='green');


# In[ ]:




