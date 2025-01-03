# import the libraris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
dataset=pd.read_csv(r"E:\3.spyder\Clasification\XGBOOST\Churn_Modelling.csv")
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Label encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
print(X)

# One hot Encodeing the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Training XGboost on the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#predicting the Test set result
y_pred=classifier.predict(X_test)

# Making the Confusion arix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias=classifier.score(X_train,y_train)
bias

variance=classifier.score(X_test,y_test)
variance
