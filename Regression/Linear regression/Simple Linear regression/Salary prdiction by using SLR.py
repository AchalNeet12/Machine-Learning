# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset=pd.read_csv(r"E:\3.spyder\linear regression\Simpal linear regression\Salary_Data.csv")

# split the dataset x and y
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# simple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X_train,y_train)

# test the model
y_pred=regressor.predict(X_test)

# visualization
plt.scatter(X_train,y_train,color ='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.tilte("Salary vs Experience(TRaining set)")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.tilte("Salary vs Experience(TRaining set)")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show

# calculate the slop
m_slop=regressor.coef_
m_slop

# calculate the intercept
c_inter = regressor.intercept_
c_inter

# predict the salary
y_15=m_slop*15+c_inter
y_15

from sklearn.metrics import r2_score,mean_squared_error
R2=r2_score(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
#MSE**(1/2)
RMSE=np.sqrt(MSE)
#accuracy_score(y_test,y_pred)#it is a regression tech
print("R-square:",R2)
print("MSE:",MSE)
print("RMSE:",RMSE)

# Regration table code
# Introduce to OLS 
from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()
