# Decision tree  algorithm

# import the libraries
import pandas as pd
import numpy as np

# load the data
dataset = pd.read_csv(r"D:\DS NOTES\Machine Learning\30th - NAIVE BAYES\Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

# split the data into training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Decisio tree classification model on the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# Making confusion matrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

# Acurracy of this model 
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias=classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance
