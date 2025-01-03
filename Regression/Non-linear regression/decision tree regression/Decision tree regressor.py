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

# Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor_dt=DecisionTreeRegressor(criterion='squared_error',splitter='random',random_state=0)
regressor_dt.fit(X,y)

y_pred_dt=regressor_dt.predict([[6.5]])
y_pred_dt
