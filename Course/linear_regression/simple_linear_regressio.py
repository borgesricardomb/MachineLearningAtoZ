#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data
dataset = pd.read_csv('../../Data/linear_regression/Salary_Data.csv')
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1 ].values

#Train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the y_pred based on the train_test
y_pred = regressor.predict(X_test)

#Plotting the regression
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_pred, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

#Predicting a single observation
y_obs = regressor.predict([[12]])

#Coeficients #AX + B
a, b = regressor.coef_, regressor.intercept_