# import the libraries
import numpy as np
import pandas as pd

#load the dataset
dataset=pd.read_csv(r"D:\DS NOTES\Machine Learning\30th - NAIVE BAYES\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# split the data in training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

# Accuracy of this model
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias=classifier.score(X_train,y_train)
bias

variance=classifier.score(X_test,y_test)
variance
