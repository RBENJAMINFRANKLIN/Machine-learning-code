

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:42:22 2019

@author: Benjamin Franklin
"""

 #importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = np.array(y).reshape(-1,1)
y = sc_y.fit_transform(y)


# Fiting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# Predicting a new result 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# visualisig the svr results
plt.scatter(X,y,color = 'red')#real observation pts
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff ( regression model)')
plt.xlabel('Position lever')
plt.ylabel('Salary')
plt.show()

# visualisig the  regression results
plt.scatter(X,y,color = 'red')#real observation pts(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(y), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff ( regression model)')
plt.xlabel('Position lever')
plt.ylabel('Salary')
plt.show()