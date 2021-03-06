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
'''from sklearn.preprocessing import Stan7dardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fiting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualisig the linear regression results
plt.scatter(X,y,color = 'red')#real observation pts
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')￼
plt.ylabel('Salary')
plt.show()


# visualisig the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')#real observation pts
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position lever')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression
lin_reg.predict(6.5)

# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))