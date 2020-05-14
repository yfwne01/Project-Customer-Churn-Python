#!/usr/bin/env python
# coding: utf-8

# In[199]:


#load packages
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[200]:


#read the dataset
customer_churn = pd.read_csv('MBRChurnModel_FirstYear_MSK (1).csv')


# In[201]:


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


# In[202]:


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


# In[203]:


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


# In[204]:


#drop irrelevant columns
churn_new.drop(['A2ACCIPK'],axis=1,inplace=True)
churn_new.drop(['ZIPCODE'],axis=1,inplace=True)


# In[ ]:





# In[205]:


df_new = churn_new
#encode the response variable into factor 1 and factor 0
df_new.replace(to_replace={"RENEW": {'Y': '1', 'N': '0'}},
           inplace=True)


# In[206]:


#convert categorical features to dummy variables (replace)
df_new.replace(to_replace={"M2EXCFLG": {'E': 1, 'N': 0}},
           inplace=True)

df_new.replace(to_replace={"HOMEFCTYCHANGE": {'Y': 1, 'N': 0}},
           inplace=True)

df_new.replace(to_replace={"RECENTMOVING": {'Y': 1, 'N': 0}},
           inplace=True)


# In[207]:


#convert categorical features to dummy variables (one-hot-encoding)
locations = pd.get_dummies(df_new['F2HOMRGN'],drop_first=True)
df_new.drop(['F2HOMRGN'],axis=1,inplace=True)
df_new = pd.concat([df_new,locations],axis=1)


# In[208]:


#check
df_new.info()


# In[209]:


df_new.shape


# In[210]:


#Feature selection
#univariate feature selection (ANOVA)(use training set only)
ftest_df = df_new


# In[211]:


#get the training and testing dataset 
from sklearn.model_selection import train_test_split
X = ftest_df.drop('RENEW',axis=1)
y = ftest_df['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500)


# In[212]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions

# feature extraction
test = SelectKBest(f_classif, k=10)
fit = test.fit(X_train, y_train)


# In[213]:


# summarize f-scores
#select features that return high f-values and use those for further analysis
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_train)


# In[214]:



#drop features with negative F-values
ftest_df.drop(['A2ACCTYP'],axis=1,inplace=True)
ftest_df.drop(['B2BUSTYP'],axis=1,inplace=True)
ftest_df.drop(['RECENTMOVING'],axis=1,inplace=True)
ftest_df.drop(['NE'],axis=1,inplace=True)
ftest_df.drop(['SD'],axis=1,inplace=True)


# In[215]:


#check
ftest_df.head()


# In[216]:


ftest_df.shape


# In[31]:


#ML Models


# In[32]:


#1.Logistic Regression (before feature selection)
from sklearn.model_selection import train_test_split

df_log = df_new
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[33]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[34]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
logmodel = LogisticRegression(random_state = 100) 
logmodel.fit(X_train, y_train) 


# In[35]:


#prediction
log_pred = logmodel.predict(X_test)
#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,log_pred))


# In[36]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, log_pred) 


# In[143]:


#get the accuarcy
a = (15752+7495)/(15752+4058+5946+7495)
a


# In[37]:


#after feature selection
from sklearn.model_selection import train_test_split

df_log = ftest_df
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[38]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[42]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
logmodel = LogisticRegressionCV(cv=10,random_state = 100) 
logmodel.fit(X_train, y_train) 

#prediction
log_pred = logmodel.predict(X_test)
#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,log_pred))


# In[43]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, log_pred) 


# In[22]:


#Recursive Feature Elimination
from sklearn.model_selection import train_test_split
df_log = df_new
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[23]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:



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


# In[25]:


df_new.info()


# In[20]:


rfe_df = df_new[['RENEW','M2EXCFLG','AGE','TENURE','EARLYFAREWELL','SHOP1YR','SHOP6M','SHOP3M','ECOMSHOP','GROCERYSHOP','MW']]


# In[21]:


rfe_df.shape


# In[28]:


rfe_df.info()


# In[74]:


#run the log model again
from sklearn.model_selection import train_test_split

df_log = rfe_df
X = df_log.drop('RENEW',axis=1)
y = df_log['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[75]:


#standardize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[76]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
logmodel = LogisticRegressionCV(cv=10,random_state = 100) 
logmodel.fit(X_train, y_train) 

#prediction
log_pred = logmodel.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,log_pred))


# In[77]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, log_pred) 


# In[217]:


#Random Forest
#after the f-test feature selection
df_rf = ftest_df


# In[218]:


df_rf.shape


# In[174]:



from sklearn.model_selection import train_test_split
X = df_rf.drop('RENEW',axis=1)
y = df_rf['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[175]:


#build the RF model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)


# In[176]:


#prediction
rf_prediction = rf.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_prediction))


# In[177]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,rf_prediction) 


# In[178]:


#get the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,rf_prediction)


# In[219]:


#tune RF
from sklearn.model_selection import train_test_split
X = df_rf01.drop('RENEW',axis=1)
y = df_rf01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[ ]:


#define models and parameters
tune_rf = RandomForestClassifier(random_state = 105)

n_estimators = [150,180,200,220,250,300,350,500,650,700]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)


# In[ ]:


#define the grid search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=tune_rf, param_grid=hyperF, n_jobs=-1, cv=3, verbose = 1,scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)


# In[ ]:


grid_result.best_params_


# In[220]:


#use those hyperparameters 
forestOpt = RandomForestClassifier(random_state = 105, max_depth = 25, n_estimators = 250, min_samples_split = 5, min_samples_leaf = 2,criterion='gini')                                  
modelOpt = forestOpt.fit(X_train, y_train)
y_pred = modelOpt.predict(X_test)


# In[221]:


#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:


#Tree-based feature selection (RF)


# In[88]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=250), max_features=10)
embeded_rf_selector.fit(X_train, y_train)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(embeded_rf_feature)


# In[179]:


#after the feature selection
df_rf01 = ftest_df[['RENEW','F2HOMFCY', 'AGE', 'DISTANCE', 'EARLYFAREWELL', 'SHOP1YR', 'SHOP6M', 'SHOP3M', 'MEDICALSHOP', 'GROCERYSHOP']]


# In[180]:


#run the RF model again 
from sklearn.model_selection import train_test_split
X = df_rf01.drop('RENEW',axis=1)
y = df_rf01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[181]:


#use specific hyperparameters (after grid search)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 105, max_depth = 25, n_estimators = 250, min_samples_split = 5, min_samples_leaf = 2,criterion='gini')
rf.fit(X_train,y_train)


# In[182]:


#prediction
rf_prediction = rf.predict(X_test)

#evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_prediction))


# In[183]:


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,rf_prediction) 


# In[184]:


#get the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,rf_prediction)


# In[ ]:





# In[ ]:


#KNN


# In[ ]:


#After f-test feature selection


# In[27]:


df01_knn = ftest_df


# In[28]:


df01_knn.shape


# In[29]:


from sklearn.model_selection import train_test_split
X = df01_knn.drop('RENEW',axis=1)
y = df01_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[30]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[31]:


#build the knn model (k=25)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25,metric='euclidean')
knn.fit(X_train,y_train)


# In[33]:


#prediction
pred = knn.predict(X_test)
#evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))


# In[ ]:





# In[34]:


#find the best k
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(25,65):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[35]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,65),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[37]:


#run the model with the best k
knn = KNeighborsClassifier(n_neighbors=63,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=63')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[185]:


#accuracy
a = (15982+7119)/(15982+3828+6322+7119)
a


# In[ ]:


#After RFE feature selection


# In[38]:


df02_knn = rfe_df


# In[39]:


df02_knn.shape


# In[40]:


from sklearn.model_selection import train_test_split
X = df02_knn.drop('RENEW',axis=1)
y = df02_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[41]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[42]:


#build the knn model
from sklearn.neighbors import KNeighborsClassifier

#find the best k
error_rate = []

for i in range(25,65):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[43]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,65),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[44]:


#run the model with the best k
knn = KNeighborsClassifier(n_neighbors=54,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=54')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[187]:


#accuracy
a = (15606+7785)/(15606+4204+5656+7785)
a


# In[ ]:





# In[98]:


#After tree-based feature selection


# In[45]:


df03_knn = df_rf01


# In[46]:


df03_knn.shape


# In[47]:


from sklearn.model_selection import train_test_split
X = df03_knn.drop('RENEW',axis=1)
y = df03_knn['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[48]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[49]:


#build the knn model
from sklearn.neighbors import KNeighborsClassifier

#find the best k
error_rate = []
for i in range(25,65):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[50]:


#plot
plt.figure(figsize=(10,5))
plt.plot(range(25,65),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[52]:


#run the model with the best k
knn = KNeighborsClassifier(n_neighbors=46,metric='euclidean')

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=46')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[188]:


#accuracy
a = (15301+7884)/(15301+7884+4509+5557)
a


# In[ ]:





# In[ ]:


#SVM


# In[19]:


#after the f-test feature selection
df_svm01 = ftest_df

from sklearn.model_selection import train_test_split
X = df_svm01.drop('RENEW',axis=1)
y = df_svm01['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[54]:



from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[55]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[56]:


print(confusion_matrix(y_test,y_pred))


# In[149]:


a = (19809+1855)/(19809+1+11586+1855)
a


# In[ ]:


#After RFE feature selection


# In[58]:


df_svm02 = rfe_df
from sklearn.model_selection import train_test_split
X = df_svm02.drop('RENEW',axis=1)
y = df_svm02['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[59]:


from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[60]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[61]:


print(confusion_matrix(y_test,y_pred))


# In[150]:


a = (19780+1863)/(19780+30+11578+1863)
a


# In[ ]:





# In[ ]:


#After tree-based feature selection


# In[25]:


df_svm03 = df_rf01
from sklearn.model_selection import train_test_split
X = df_svm03.drop('RENEW',axis=1)
y = df_svm03['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[64]:


from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(kernel='rbf') # radial kernel

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[65]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[139]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


#Tune SVM (Grid search)


# In[42]:


#after tree-based feature selection
df_svm03 = df_rf01
from sklearn.model_selection import train_test_split
X = df_svm03.drop('RENEW',axis=1)
y = df_svm03['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[43]:


#define the tuning parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}


# In[44]:


from sklearn.svm import SVC
from sklearn import svm
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)


# In[45]:


grid.best_params_


# In[46]:


print(grid.best_estimator_)


# In[134]:


from sklearn.svm import SVC
from sklearn import svm

#create a svm Classifier
clf = svm.SVC(C=1,cache_size=200,class_weight=None,coef0=0.0,
    decision_function_shape='ovr',degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=True,random_state=105,shrinking=True,tol=0.001,
    verbose=False)

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[135]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[ ]:


#ROC


# In[141]:


from sklearn.metrics import roc_curve, roc_auc_score
probs = clf.predict_proba(X_test)


# In[151]:


#keep the prob for positive class
probs = probs[:, 1]
#get the AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:





# In[ ]:


#PCA


# In[97]:


df_pca = df_new
df_pca['RENEW'].replace('0', 'Churn',inplace=True)
df_pca['RENEW'].replace('1', 'Non_Churn',inplace=True)


# In[98]:


df_pca.head()


# In[99]:


from sklearn.model_selection import train_test_split
X = df_pca.drop('RENEW',axis=1)
y = df_pca['RENEW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 500) 


# In[100]:


#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[101]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[102]:


print(pca.components_)


# In[103]:


print(pca.explained_variance_)


# In[ ]:





# In[127]:


#SVM
from sklearn.svm import SVC
from sklearn import svm

clf = svm.SVC(probability=True) 

#train the model using the training sets
clf.fit(X_train, y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


# In[128]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[129]:


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


# In[119]:


pca = PCA(n_components=2)
pca.fit(X_pca)
X_pca = pca.transform(X_pca)


# In[121]:


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[130]:


#get the ROC of SVM+PCA
from sklearn.metrics import roc_curve, roc_auc_score
probs = clf.predict_proba(X_test)


# In[131]:


#keep the prob for positive class
probs = probs[:, 1]
#get the AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:




