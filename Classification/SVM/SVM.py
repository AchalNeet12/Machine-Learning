# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read the dataset
dataset=pd.read_csv(r"E:\3.spyder\Clasification\logistic regression\logit classification.csv")

# load the dataset in x and y
X= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,1].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Training the SM model on the Training set
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

# get the model accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias = classifier.score(X_train,y_train)
bias

variance=classifier.score(X_test,y_test)
variance

# get the classificatio report
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
