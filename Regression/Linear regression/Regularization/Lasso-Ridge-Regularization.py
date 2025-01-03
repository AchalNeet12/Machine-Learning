# import numerical libraries
import numpy as np
import pandas as pd

#importing graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#importing linear regression machine learning libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score

data=pd.read_csv(r"E:\3.spyder\linear regression\regularization\car-mpg.csv")
data.head()

# Drop car name
#Replace origin into 1,2,3.. dont forget get_dummies
#Replace? with nan
#Replace all nan with median
data=data.drop(['car_name'],axis=1)
data['origin']=data['origin'].replace({1:'america',2:'europe',3:'asia'})
data=pd.get_dummies(data,columns=['origin'],dtype=int)
data=data.replace('?',np.nan)
data=data.apply(lambda x:x.fillna(x.median()),axis=0)

X=data.drop(['mpg'],axis=1)# independent variable
y=data[['mpg']] # dependent variable

#Scaling the data
X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s,columns=X.columns)#convering scaled data into dataframe

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s,columns=y.columns)#ideally train, test data should be in columns

#split the data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape

# simple linear model
#fit simpal linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train,y_train)
for idx,col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name,regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]
print("The intercept is {}".format(intercept))

# Regularized Ridge Regression

# alpha factor here is lambda (penalty term) which helps toreduce the magnitude of coeff

ridge_model  = Ridge(alpha=0.3)
ridge_model.fit(X_train,y_train)

print("Ridge model coef:{}".format(ridge_model.coef_))
# As the data has 10 columns hence 10 coefficient appear here

# Regularized Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

print("Lasso model coef:{}".format(ridge_model.coef_))
# As the data has 10 columns hence 10 coefficient appear here

# score Comparison
#Model core-r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS)=Regression error/TSS

#Simple Linear Model
print(regression_model.score(X_train,y_train))
print(regression_model.score(X_test,y_test))

print("*********************")
#Ridge
print(ridge_model.score(X_train,y_train))
print(ridge_model.score(X_test,y_test))

print("**************************")
#Lasso
print(lasso_model.score(X_train,y_train))
print(lasso_model,score(X_test,y_test))


