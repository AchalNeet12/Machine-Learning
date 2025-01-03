# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset 
dataset=pd.read_csv(r"E:\3.spyder\Regression\linear regression\Multiple linear regressin\House_data.csv")

# load the data in x and y variable
X = dataset.iloc[:, [3, 4, 5, 7, 9, 14, 19]]
y = dataset.iloc[:, 2]  # 2nd column (dependent variable)

#fit the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# test the model
y_pred=regressor.predict(X_test)

# calculate the slop
m_slop=regressor.coef_
m_slop

# calculate the intercept
c_inter = regressor.intercept_
c_inter

