# Simple Frame Work for building logistic regression using python

# Importing various useful libraries in python 
import numpy as np
import matplotlib as plt

# using sklearn package (scikit learn) for easy usage 
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression 

# In[3]:
print(datasets)

# In[4]:
dataset = datasets.load_iris()
dataset

# In[6]:
model = LogisticRegression()
model.fit(dataset.data,dataset.target) # x is the training vector or explanatory variable , y is the target vector or response variable

# In[7]:
expected = dataset.target
predicted = model.predict(dataset.data)


# In[8]:
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
