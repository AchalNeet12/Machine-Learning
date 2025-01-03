import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load the dataset
dataset=pd.read_csv(r"E:\3.spyder\Regression\non-linear regression\Polynomial regression\emp_sal.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='poly',degree=5)
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
















