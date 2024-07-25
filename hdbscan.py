#!/usr/bin/env python
# coding: utf-8

# # Session 2: Hierarchical Clustering and DBSCAN

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Accessing dataset

# In[2]:


df=pd.read_excel('Allianz.xlsx',sheet_name=1)
df


# In[3]:


df.info()


# ## Preprocessing

# In[4]:


df.columns


# In[6]:


df_1=df.drop(['Contract_number','Postalcode', 'District'],axis=1)
df_1


# In[7]:


df_2=df_1.dropna()
df_2


# In[8]:


# Standardising

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_scaled=scaler.fit_transform(df_2)
X_scaled


# In[9]:


df_final=pd.DataFrame(X_scaled,columns=df_2.columns)
df_final


# ## Implementing Hierarchical clustering

# In[10]:


from scipy.cluster.hierarchy import dendrogram, linkage,fcluster


# In[11]:


link=linkage(df_final,method='ward')
link


# In[12]:


plt.figure(figsize=(12,8))
dendrogram(link)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Dendrogram');


# In[13]:


hier_labels=fcluster(link, t=200,criterion='distance')
hier_labels


# In[14]:


df_2['Hier_cluster']=hier_labels


# In[15]:


df_2


# In[16]:


df_2['Hier_cluster'].value_counts()


# In[17]:


df_2[(df_2['Sex']==1) & (df_2['Hier_cluster']==1)]


# In[18]:


df_2


# ## Implementing DBSCAN

# In[24]:


from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=0.7,min_samples=5)
dbscan


# In[25]:


db_labels=dbscan.fit_predict(df_final)
db_labels


# In[26]:


df_2['DB_cluster']=db_labels


# In[27]:


df_2


# In[28]:


df_2['DB_cluster'].value_counts()


# In[ ]:





# In[ ]:




