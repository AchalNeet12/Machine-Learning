import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load the dataset
dataset=pd.read_csv(r"E:\3.spyder\non-linear regression\Polynomial regression\emp_sal.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# linear model --linear algor(degree-1)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Random forest
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(n_estimators=50)
reg_rf.fit(X,y)
 
y_pred_rf=reg_rf.predict([[6.5]])
y_pred_rf
 