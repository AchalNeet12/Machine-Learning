# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv(r"E:\3.spyder\Clasification\Social_Network_Ads.csv")
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

# FEture Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Training the kernal sVM model on the Training set
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)

# Predicting the test set result
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#Accuracy or the model
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance

# Appling the K-fold  cross validation
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=5)
print("Accuracy:{:.2f}%".format(accuracy.mean()*100))
print("Standard Deviation:{:.2f}%".format(accuracy.std()*100))

# Appling Grid Search to find best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1, 10, 100, 1000],'kernel':['linear']},
            {'C':[1, 10, 100, 1000],'kernel':['rbf'],'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           )
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Acccuracy:{:.2f}%".format(best_accuracy*100))
print("Best Parameters:",best_parameters)


# Randon search cv
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distributions={
    'n_estimators': randint(20,200),
    'max_depth':randint(3,10)
    }
random_search = RandomizedSearchCV(estimator=classifier,param_distributions=param_distributions,n_iter=10,cv=5)
random_search.fit(X_train,y_train)
