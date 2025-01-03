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

# linear regression  visualization
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Polymonial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# it is not a best graph we do the hyper parameter tunning so we fit the dgree

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred






















