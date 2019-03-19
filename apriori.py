# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:09:34 2019

@author: Benjamin Franklin
"""

 #importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,min_confidence = 0.2,min_lift=3,min_length=2)

#visualising the results
results = list(rules)

clean_results = []
for i in range(0, len(results)):
    clean_results.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nCONFIDENCE:\t' + str(results[i][2][0][2]) + '\nLIFT:\t' + str(results[i][2][0][3]))