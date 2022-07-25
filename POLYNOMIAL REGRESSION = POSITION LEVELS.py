# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 20:41:27 2022

@author: Asim Pardeshi
"""
#IMPORTING THE PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATASET
dataset = pd.read_csv(r'D:\GCLASSROOM\18july\18th\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')

#SPLIT DATASET INTO DEPENDENT AND INDEPENDENT VARIABLE
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

lin_reg.fit(x,y)

#FITTING POLYNOMIAL REGRESSION TO THE DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#PLOTTING BY TRUE OBSERVATION
plt.scatter(x,y,color='red') #WE ARE GOING TO PLOT FOR ACTUAL VALUE OF X & Y
plt.plot(x,lin_reg.predict(x),color='blue') #NOW PLOT FOR THE PREDICTION LINE WHERE X COORDINATE PREDICTED
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#VISUALIZING THE POLYNOMIAL RESULT
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Trurth or Bluff(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#PREDICTING A NEW RESULT WITH LINEAR REGRESSION
lin_reg.predict([[6.5]]) #SLR

#PREDICTING A NEW RESULT WITH POLYNOMIAL REGRESSION
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
