#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# # importing the dataset

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Heart disease prediction/heart.csv')


# In[3]:


#checking the first 5 row of the data
df.head()


# In[4]:


#5 row from the end of the data
df.tail()


# In[5]:


#size of the dataset
df.shape


# In[6]:


#getting some info of the data
df.info()


# In[7]:


#checking for any null value
df.isnull().sum()


# In[8]:


#to get the statistical measure of the data
df.describe()


# In[9]:


#to see the mean of data across the target
df.groupby(['target']).mean()


# In[10]:


#checking the distribution target of target variable
'''
1 ---> ill heart
2 ---> healthy heart
'''
df['target'].value_counts()


# # slitting the features and label

# In[11]:


X = df.drop('target', axis =1)
Y = df['target']


# In[12]:


print(X)
print(Y)


# # splitting the data into train and tets data

# In[13]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


# In[14]:


#shape of the train and tets data
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# # model training

# In[15]:


model = LogisticRegression()


# In[16]:


#training the logistic regression model
model.fit(x_train, y_train)


# # Model evaluation

# In[17]:


#accuracy score


# In[18]:


#on training data
training_pred = model.predict(x_train)

train_acc_score = accuracy_score(training_pred, y_train)
print('TRAINING ACCURACY IS :', train_acc_score)


# In[19]:


#on testing data
testing_pred = model.predict(x_test)

test_acc_score = accuracy_score(testing_pred, y_test)
print('TESTING ACCURACY IS :', test_acc_score)


# # Building a predictive system

# In[20]:


model_input= input()

input_list = [float(i) for i in model_input.split(',')]
input_array= np.asarray(input_list)
#reshaping the array
input_array = input_array.reshape(1, -1)

prediction = model.predict(input_array)
print('THE PREDICTION IS :', prediction)
if prediction == 1:
    print('THE PERSON IS IN CRITICAL CONDITION')
else:
    print('THE PERSON IS HEALTHY')


# In[ ]:




