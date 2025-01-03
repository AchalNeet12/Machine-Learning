# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# imporing the dataset
dataset = pd.read_csv(r"E:\3.spyder\Clasification\Social_Network_Ads.csv")

# splitting dataset into x and y
X = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,-1]

#sptting the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the k-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
 
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

# this is to get the model Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

bias = classifier.score(X_train,y_train)
bias

variance=classifier.score(X_test,y_test)
variance

# get the classificatio report
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
