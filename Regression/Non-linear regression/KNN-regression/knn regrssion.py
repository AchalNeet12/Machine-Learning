import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#load the data
dataset=pd.read_csv(r"E:\3.spyder\non-linear regression\Polynomial regression\emp_sal.csv")

# split the data in x and y variable
X= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting the knn model to the dataset
from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=5,p=1,weights='distance',algorithm='brute')
regressor_knn.fit(X,y)

y_pred_knn=regressor_knn.predict([[6.5]])
y_pred_knn

