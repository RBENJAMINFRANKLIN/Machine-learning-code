
# coding: utf-8

# In[3]:


# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[6]:



# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Benjamin Franklin\Desktop\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 8 - Decision Tree Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# In[8]:


# Fitting Decision Tree Regression to the dataset using library defaults
from sklearn.tree import DecisionTreeRegressor
#set random state for reproducibility
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# In[9]:


# Predicting a new result
y_pred = regressor.predict(6.5)


# In[10]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

