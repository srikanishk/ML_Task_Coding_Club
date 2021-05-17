#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('task2.csv')


# In[2]:


def recode_empty_cells(dataframe, list_of_columns):                         #Replacing empty cells with 0s

    for column in list_of_columns:
        dataframe[column] = dataframe[column].replace(r'\s+', 0, regex=True)

    return dataframe

recode_empty_cells(df,df.columns)


# In[5]:


cols = list(df.columns)                                                          #forming dependant and independant data sets
x_data = df.drop(columns=['Segment type','Segment Description','Answer','It became a relationship'])
y_data = df[cols[6]]


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20)  #splitting data


# In[44]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred_svc = svclassifier.predict(X_test)
print(accuracy_score(y_test,y_pred_svc)*100)


# In[45]:


from sklearn.svm import LinearSVC

lin_clf = LinearSVC(random_state=1000)          #Changing hyperparameter for SVC
                                                #Manual Alteration of parameter
lin_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred_svc_hp = lin_clf.predict(X_train)
print(accuracy_score(y_train, y_pred_svc_hp)*100)


# In[46]:


model = LogisticRegression()
model.fit(X_test, y_test)
y_pred_LR = model.predict(X_test)
print(accuracy_score(y_test,y_pred_LR)*100)


# In[47]:


model_new = LogisticRegression(solver='lbfgs', random_state=10000, max_iter=5000)  #changing parameters for Logistic Regression
                                                                                   #Manual Alteration of paramters
model_new.fit(X_test, y_test)                                                      
y_pred_LR_hp = model_new.predict(X_test)
print(accuracy_score(y_test,y_pred_LR_hp)*100)


# In[39]:





# In[42]:




