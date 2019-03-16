
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
X_test = sc_X.transform(X_test)
sc_y = StandardSccaler()
y_train = sc_y.fit_transform(y_train)'''


# Fiting regression to the dataset
# create your regressor here



# Predicting a new result 
y_pred = regressor.predict(6.5)

# visualisig the  regression results
plt.scatter(X,y,color = 'red')#real observation pts
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff ( regression model)')
plt.xlabel('Position lever')
plt.ylabel('Salary')
plt.show()

# visualisig the  regression results
plt.scatter(X,y,color = 'red')#real observation pts(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff ( regression model)')
plt.xlabel('Position lever')
plt.ylabel('Salary')
plt.show()