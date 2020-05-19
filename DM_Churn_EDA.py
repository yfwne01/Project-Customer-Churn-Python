#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load packages
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the dataset
customer_churn = pd.read_csv('MBRChurnModel_FirstYear_MSK (1).csv')


# In[3]:


#check data information
customer_churn.shape


# In[4]:


customer_churn.head()


# In[5]:


customer_churn.info()
#4 categorical variables (predictors)


# In[28]:


#get the statistics of all numerical variables
customer_churn.describe().transpose()
#outliers exist


# In[7]:


#checked missing/nulls
customer_churn.isnull().sum()


# In[7]:


#check the duplicated rows (customer ID)
customer_churn.A2ACCIPK.duplicated().sum()


# In[8]:


#The overall distribution of churn and non-churn customers
#data
labels=customer_churn['RENEW'].value_counts().index
values=customer_churn['RENEW'].value_counts().values

plt.figure(figsize = (10, 5))
ax = sns.barplot(x=labels, y=values)
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1, values[i],ha="center")
    plt.xlabel('RENEW')
    plt.ylabel('Count')


# In[11]:


#get the pairplot of the dataset
plt.figure(figsize=(50,50))
sns.pairplot(customer_churn,hue='RENEW',palette='husl')


# In[9]:


#correlation between numeric variables
plt.figure(figsize=(15, 15))
sns.heatmap(customer_churn.corr(), annot=True,cmap='coolwarm')
plt.title('Correlation between numeric variables')
#From the correlation plot, we can see that:
#B2B and A2A are correlated;(0.86)
#Shop1yr and shop6m are correlated;(0.96)
#Shop1yr and shop3m are correlated ;(0.93)
#shop6m and shop3m are correlated;(0.97)


# In[12]:


#identify outliers
df=customer_churn
sns.boxplot(df['SHOP1YR'])


# In[13]:


sns.boxplot(df['SHOP6M'])


# In[14]:


sns.boxplot(df['SHOP3M'])


# In[16]:


#deal with outliers
#Interquartile Range(IQR) Method

Q1=df['SHOP1YR'].quantile(0.25)
Q3=df['SHOP1YR'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

#remove outliers
churn_new = df[df['SHOP1YR']< Upper_Whisker]

#check
sns.boxplot(churn_new['SHOP1YR'])


# In[17]:


Q1=churn_new['SHOP6M'].quantile(0.25)
Q3=churn_new['SHOP6M'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

#remove outliers
churn_new = df[df['SHOP6M']< Upper_Whisker]

#check
sns.boxplot(churn_new['SHOP6M'])


# In[18]:


Q1=churn_new['SHOP3M'].quantile(0.25)
Q3=churn_new['SHOP3M'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

#remove outliers
churn_new = df[df['SHOP3M']< Upper_Whisker]

#check
sns.boxplot(churn_new['SHOP3M'])


# In[49]:


#EDA
#hist for numerical variables (SHOP1YR)(6M and 3M are similiar as 1YR distribution)
df=churn_new
plt.figure(figsize=(15,10))
df[df['RENEW']=='Y']['SHOP1YR'].hist(alpha=0.5,color='red',bins=50,label='shop1yr.churn')
df[df['RENEW']=='N']['SHOP1YR'].hist(alpha=0.5,color='blue',bins=50,label='shop1yr.nonchurn')
plt.legend()
plt.xlabel('Total shopping amount in 1 year')


# In[50]:


#hist for numerical variables (DISTANCE)
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['DISTANCE'].hist(alpha=0.5,color='red',bins=50,label='distance.churn')
df[df['RENEW']=='N']['DISTANCE'].hist(alpha=0.5,color='blue',bins=50,label='distance.nonchurn')
ax.set_xlim(1,100)
plt.legend()
plt.xlabel('Distance')
plt.ylabel('Count')


# In[51]:


#hist for numerical variables (EARLYFAREWELL)
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['EARLYFAREWELL'].hist(alpha=0.5,color='red',bins=50,label='number of days not shop.churn')
df[df['RENEW']=='N']['EARLYFAREWELL'].hist(alpha=0.5,color='blue',bins=50,label='number of days not shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('Number of days not shop')
plt.ylabel('Count')


# In[52]:


#AGE
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['AGE'].hist(alpha=0.5,color='red',bins=50,label='Customer`s Age not shop.churn')
df[df['RENEW']=='N']['AGE'].hist(alpha=0.5,color='blue',bins=50,label='Customer`s Age not shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('Customer`s Age')
plt.ylabel('Count')


# In[53]:


#F2HOMFCY
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['F2HOMFCY'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['F2HOMFCY'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('F2HOMFCY')
plt.ylabel('Count')


# In[46]:


#TENURE 
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['TENURE'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['TENURE'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('TENURE')
plt.ylabel('Count')


# In[44]:


#B2BUSTYP 
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['B2BUSTYP'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['B2BUSTYP'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('B2BUSTYP')
plt.ylabel('Count')


# In[43]:


#MBRCOUNT
df=churn_new
plt.figure(figsize=(12,10))
df[df['RENEW']=='Y']['MBRCOUNT'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['MBRCOUNT'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('MBRCOUNT')
plt.ylabel('Count')


# In[42]:


#ECOMSHOP         
df=churn_new
plt.figure(figsize=(12,10))
df[df['RENEW']=='Y']['ECOMSHOP'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['ECOMSHOP'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('ECOMSHOP')
plt.ylabel('Count')    


# In[47]:


#GASSHOP
df=churn_new
plt.figure(figsize=(12,10))
df[df['RENEW']=='Y']['GASSHOP'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['GASSHOP'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('GASSHOP')
plt.ylabel('Count')   


# In[41]:


#MEDICALSHOP
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['MEDICALSHOP'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['MEDICALSHOP'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('MEDICALSHOP')
plt.ylabel('Count')   


# In[48]:


#GROCERYSHOP   
df=churn_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='Y']['GROCERYSHOP'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='N']['GROCERYSHOP'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
ax.set_xlim(1,500)
plt.legend()
plt.xlabel('GROCERYSHOP')
plt.ylabel('Count')   


# In[20]:


#boxplox for categorical variables
df=churn_new
sns.set(style="darkgrid")
total = float(len(customer_churn))
ax = sns.countplot(y="M2EXCFLG", hue="RENEW", data=df) 
plt.title('Distribution of M2EXCFLG')
plt.xlabel('Count')
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()


# In[21]:


#boxplox for categorical variables
df=churn_new
plt.figure(figsize = (10, 8))
sns.set(style="darkgrid")
total = float(len(customer_churn))
ax = sns.countplot(y="F2HOMRGN", hue="RENEW", data=df) 
plt.title('Distribution of Region')
plt.xlabel('Count')
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()


# In[22]:


#boxplox for categorical variables
df=churn_new
plt.figure(figsize = (10, 5))
sns.set(style="darkgrid")
total = float(len(customer_churn))
ax = sns.countplot(y="HOMEFCTYCHANGE", hue="RENEW", data=df) 
plt.title('Distribution of HOMEFCTYCHANGE')
plt.xlabel('Count')
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()


# In[23]:


#boxplox for categorical variables
df=churn_new
plt.figure(figsize = (10, 5))
sns.set(style="darkgrid")
total = float(len(customer_churn))
ax = sns.countplot(y="RECENTMOVING", hue="RENEW", data=df) 
plt.title('Distribution of RECENTMOVING')
plt.xlabel('Count')
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()


# In[ ]:





# In[ ]:




