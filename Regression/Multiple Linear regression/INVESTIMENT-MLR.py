# import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
dataset=pd.read_csv(r"E:\3.spyder\Regression\linear regression\Multiple linear regressin\Investment.csv")

# split the data into x and y variable
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

X = pd.get_dummies(X,dtype=int)

# fit the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

#------MLR-MODEL-------
m=regressor.coef_
m

c=regressor.intercept_
c

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)


import statsmodels.api as sm
x_opt = X[:,[0,1,2,3,4,5]]
# OLS Ordinary Least Square
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
 



import statsmodels.api as sm
x_opt = X[:,[0,1,2,3,5]]
# OLS Ordinary Least Square
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
  

import statsmodels.api as sm
x_opt = X[:,[0,1,2,3]]
# OLS Ordinary Least Square
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
 


import statsmodels.api as sm
x_opt = X[:,[0,1,3,]]
# OLS Ordinary Least Square
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
 

import statsmodels.api as sm
x_opt = X[:,[0,1]]
# OLS Ordinary Least Square
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
 

# invest the money is digital marketting








