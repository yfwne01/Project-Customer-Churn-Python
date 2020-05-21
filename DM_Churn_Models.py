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
customer_churn


# In[3]:


#deal with outliers
#Interquartile Range(IQR) Method
df=customer_churn
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


# In[4]:


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


# In[5]:


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


# In[6]:


#drop irrelevant columns
churn_new.drop(['A2ACCIPK'],axis=1,inplace=True)
churn_new.drop(['ZIPCODE'],axis=1,inplace=True)


# In[7]:


#A2ACCTYP
churn_new.drop(['A2ACCTYP'],axis=1,inplace=True)


# In[ ]:





# In[8]:


df_new = churn_new
#encode the response variable into factor 1 and factor 0
df_new.replace(to_replace={"RENEW": {'Y': '1', 'N': '0'}},
           inplace=True)


# In[9]:


df_new.info()


# In[10]:


#From EDA (drop the numerical features that can not help c;assifiy the churn and non-churn customers)
df_new.drop(['B2BUSTYP'],axis=1,inplace=True)
df_new.drop(['ECOMSHOP'],axis=1,inplace=True)
df_new.drop(['GASSHOP'],axis=1,inplace=True)
df_new.drop(['MEDICALSHOP'],axis=1,inplace=True)


# In[11]:


#use the log +1 transformation 
#handle the skewed numerical features
df_new['SHOP1YR']= df_new['SHOP1YR']+1
df_new['SHOP6M'] = df_new['SHOP6M'] +1
df_new['SHOP3M'] = df_new['SHOP3M'] +1


# In[12]:


#SHOP1YR;6M;3M
df_new['SHOP1YR'] = np.log(df_new['SHOP1YR'])
df_new['SHOP6M'] = np.log(df_new['SHOP6M'])
df_new['SHOP3M'] = np.log(df_new['SHOP3M'])


# In[13]:


#DISTANCE
df_new['DISTANCE'] = np.log(df_new['DISTANCE'])


# In[14]:


#EARLYFAREWELL
df_new['EARLYFAREWELL'] = np.log(df_new['EARLYFAREWELL'])


# In[15]:


#check the distribution
df=df_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='1']['EARLYFAREWELL'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='0']['EARLYFAREWELL'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
plt.legend()
plt.xlabel('EARLYFAREWELL')
plt.ylabel('Count')


# In[16]:


df=df_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='1']['DISTANCE'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='0']['DISTANCE'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
plt.legend()
plt.xlabel('DISTANCE')
plt.ylabel('Count')


# In[17]:


df=df_new
plt.figure(figsize=(10,10))
df[df['RENEW']=='1']['SHOP1YR'].hist(alpha=0.5,color='red',bins=50,label='shop.churn')
df[df['RENEW']=='0']['SHOP1YR'].hist(alpha=0.5,color='blue',bins=50,label='shop.nonchurn')
plt.legend()
plt.xlabel('SHOP1YR')
plt.ylabel('Count')


# In[ ]:





# In[18]:


#convert categorical features to dummy variables (replace)
df_new.replace(to_replace={"M2EXCFLG": {'E': 1, 'N': 0}},
           inplace=True)

df_new.replace(to_replace={"HOMEFCTYCHANGE": {'Y': 1, 'N': 0}},
           inplace=True)

df_new.replace(to_replace={"RECENTMOVING": {'Y': 1, 'N': 0}},
           inplace=True)


# In[19]:


#convert categorical features to dummy variables (one-hot-encoding)
locations = pd.get_dummies(df_new['F2HOMRGN'],drop_first=True)
df_new.drop(['F2HOMRGN'],axis=1,inplace=True)
df_new = pd.concat([df_new,locations],axis=1)


# In[20]:


#check
df_new.info()


# In[21]:


df_new.shape


# In[22]:


df_new.isnull().sum()


# In[23]:


#Feature selection
#univariate feature selection (ANOVA)(use training set only)
ftest_df = df_new


# In[24]:


#get the training and testing dataset 
from sklearn.model_selection import train_test_split
X = ftest_df.drop('RENEW',axis=1)
y = ftest_df['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500)


# In[25]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions

# feature extraction
test = SelectKBest(f_classif, k=10)
fit = test.fit(X_train, y_train)


# In[26]:


# summarize f-scores
#select features that return high f-values and use those for further analysis
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_train)


# In[27]:


#drop features with negative F-values
ftest_df.drop(['HOMEFCTYCHANGE'],axis=1,inplace=True)
ftest_df.drop(['SD'],axis=1,inplace=True)
ftest_df.drop(['NE'],axis=1,inplace=True)


# In[28]:


#check
ftest_df.head()


# In[31]:


ftest_df.shape


# In[30]:


#ML Models


# In[32]:


#logistic Regression
from sklearn.model_selection import train_test_split
df_log = ftest_df
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[252]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[253]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
logmodel = LogisticRegressionCV(cv=10,random_state = 100) 
logmodel.fit(X_train, y_train) 

#prediction
log_pred = logmodel.predict(X_test)
#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,log_pred))


# In[254]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, log_pred) 


# In[31]:


#Recursive Feature Elimination
from sklearn.model_selection import train_test_split
df_log = ftest_df
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[256]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[257]:



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 10 attributes
rfe = RFE(model,10)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# In[258]:


df_log.info()


# In[33]:


rfe_df = df_log[['RENEW','F2HOMFCY','AGE','TENURE','DISTANCE','RECENTMOVING','SHOP1YR','SHOP6M','SHOP3M','MW']]


# In[33]:


rfe_df.shape


# In[261]:


rfe_df.info()


# In[263]:


#run the log model again
from sklearn.model_selection import train_test_split
df_log = rfe_df
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[264]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[265]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
logmodel = LogisticRegressionCV(cv=10,random_state = 100) 
logmodel.fit(X_train, y_train) 

#prediction
log_pred = logmodel.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,log_pred))


# In[266]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, log_pred) 


# In[ ]:





# In[113]:


#Random Forest
#after the f-test feature selection
df_rf = ftest_df


# In[114]:


df_rf.shape


# In[115]:



from sklearn.model_selection import train_test_split
X = df_rf.drop('RENEW',axis=1)
y = df_rf['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[116]:


#build the RF model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=800)
rf.fit(X_train,y_train)


# In[109]:


#prediction
rf_prediction = rf.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_prediction))


# In[117]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,rf_prediction) 


# In[118]:


#get the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,rf_prediction)


# In[119]:


#get the AUC
from sklearn.metrics import roc_curve, roc_auc_score
probs = rf.predict_proba(X_test)
#keep the prob for positive class
probs = probs[:, 1]
#get the AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:





# In[120]:


#tune RF
from sklearn.model_selection import train_test_split
df_rf01 = ftest_df
X = df_rf01.drop('RENEW',axis=1)
y = df_rf01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 105) 


# In[121]:


#define models and parameters
tune_rf = RandomForestClassifier(random_state = 105)

n_estimators = [200,220,300,350,400,450,500,600,650,700]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 20,50]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)


# In[122]:


#define the grid search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=tune_rf, param_grid=hyperF, n_jobs=-1, cv=3, verbose = 2,scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)


# In[123]:


grid_result.best_params_


# In[125]:


#use those hyperparameters 
forestOpt = RandomForestClassifier(random_state = 105, max_depth = 30, n_estimators = 400, min_samples_split = 2, min_samples_leaf = 2,criterion='gini')                                  
modelOpt = forestOpt.fit(X_train, y_train)
y_pred = modelOpt.predict(X_test)


# In[126]:


#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[127]:


#get the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,y_pred)


# In[ ]:


#Tree-based feature selection (RF)


# In[156]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=800), max_features=8)
embeded_rf_selector.fit(X_train, y_train)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(embeded_rf_feature)


# In[143]:


#after the feature selection
df_rf01 = ftest_df[['RENEW','F2HOMFCY', 'AGE', 'DISTANCE', 'EARLYFAREWELL', 'SHOP1YR', 'SHOP6M', 'SHOP3M', 'GROCERYSHOP']]


# In[144]:


from sklearn.model_selection import train_test_split
X = df_rf01.drop('RENEW',axis=1)
y = df_rf01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[145]:


#build the RF model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)


# In[146]:


#prediction
rf_prediction = rf.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_prediction))


# In[ ]:





# In[147]:


#use specific hyperparameters (after grid search)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 105, max_depth = 30, n_estimators = 400, min_samples_split = 2, min_samples_leaf = 2,criterion='gini')
rf.fit(X_train,y_train)


# In[148]:


#prediction
rf_prediction = rf.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_prediction))


# In[149]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,rf_prediction) 


# In[150]:


#get the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,rf_prediction)


# In[ ]:





# In[ ]:


#KNN


# In[ ]:


#After f-test feature selection


# In[80]:


df01_knn = ftest_df


# In[81]:


df01_knn.shape


# In[82]:


from sklearn.model_selection import train_test_split
X = df01_knn.drop('RENEW',axis=1)
y = df01_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[83]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[293]:


#build the knn model (k=25)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25,metric='euclidean')
knn.fit(X_train,y_train)


# In[294]:


#prediction
pred = knn.predict(X_test)
#evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))


# In[ ]:





# In[84]:


#find the best k
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(25,65):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[85]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,65),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[86]:


#run the model with the best k
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=61,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print('WITH K=61')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


#After RFE feature selection


# In[87]:


df02_knn = rfe_df


# In[88]:


df02_knn.shape


# In[89]:


from sklearn.model_selection import train_test_split
X = df02_knn.drop('RENEW',axis=1)
y = df02_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[90]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[91]:


#build the knn model
from sklearn.neighbors import KNeighborsClassifier

#find the best k
error_rate = []

for i in range(25,65):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[92]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,65),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[93]:


#run the model with the best k
knn = KNeighborsClassifier(n_neighbors=62,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=62')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:





# In[98]:


#After tree-based feature selection


# In[98]:


df03_knn = df_rf01


# In[99]:


df03_knn.shape


# In[100]:


from sklearn.model_selection import train_test_split
X = df03_knn.drop('RENEW',axis=1)
y = df03_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[101]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[102]:


#build the knn model
from sklearn.neighbors import KNeighborsClassifier

#find the best k
error_rate = []
for i in range(25,80):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[103]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,80),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[104]:


#run the model with the best k
knn = KNeighborsClassifier(n_neighbors=70,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=70')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:


#SVM


# In[43]:


#after the f-test feature selection
df_svm01 = ftest_df

from sklearn.model_selection import train_test_split
X = df_svm01.drop('RENEW',axis=1)
y = df_svm01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[44]:



from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[196]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


#After RFE feature selection


# In[46]:


df_svm02 = rfe_df
from sklearn.model_selection import train_test_split
X = df_svm02.drop('RENEW',axis=1)
y = df_svm02['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[47]:


from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[49]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[ ]:


#After tree feature selection


# In[50]:


df_svm03 = df_rf01
from sklearn.model_selection import train_test_split
X = df_svm03.drop('RENEW',axis=1)
y = df_svm03['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[51]:


from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[52]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[54]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[ ]:


#Tune SVM (Grid search)


# In[36]:


#after f test feature selection
df_svm03 = ftest_df
from sklearn.model_selection import train_test_split
X = df_svm03.drop('RENEW',axis=1)
y = df_svm03['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[37]:


#define the tuning parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}


# In[38]:


from sklearn.svm import SVC
from sklearn import svm
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)


# In[39]:


grid.best_params_


# In[40]:


print(grid.best_estimator_)


# In[41]:


grid_predictions = grid.predict(X_test)


# In[43]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, grid_predictions))


# In[ ]:





# In[ ]:


#PCA


# In[70]:


df_pca = ftest_df
df_pca['RENEW'].replace('0', 'Churn',inplace=True)
df_pca['RENEW'].replace('1', 'Non_Churn',inplace=True)


# In[71]:


df_pca.head()


# In[72]:


from sklearn.model_selection import train_test_split
X = df_pca.drop('RENEW',axis=1)
y = df_pca['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[73]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[74]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[75]:


print(pca.components_)


# In[76]:


print(pca.explained_variance_)


# In[ ]:





# In[77]:


#SVM
from sklearn.svm import SVC
from sklearn import svm

clf = svm.SVC(probability=True) 

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[78]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[79]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,y_pred)


# In[ ]:





# In[ ]:


#PCA visualization
X = df_pca.drop('RENEW',axis=1)
y = df_pca['RENEW']


# In[117]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_pca = scaler.transform(X)


# In[58]:


pca = PCA(n_components=6)
pca.fit(X_pca)
X_pca = pca.transform(X_pca)


# In[121]:


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[56]:


#get the ROC of SVM+PCA
from sklearn.metrics import roc_curve, roc_auc_score
probs = clf.predict_proba(X_test)


# In[57]:


#keep the prob for positive class
probs = probs[:, 1]
#get the AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:




