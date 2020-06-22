#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

#Import data
dataset = pd.read_csv('../../Data/multilinar_regression/50_Startups.csv')
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1 ].values

#Data Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) #Importante: Aqui não precisa dropar uma Dummy para RODAR, mas sim para fazer a BACKWARD ELIMINATION

#Train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the y_pred based on the train_test
y_pred = regressor.predict(X_test)

#Evaluating model
from sklearn.metrics import r2_score
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis = 1))
r2_score(y_test, y_pred)

#Predicting a single observation
y_obs = regressor.predict([[1, 0, 0, 160000, 130000, 300000]])

#Coeficients #AX + B
a, b = regressor.coef_, regressor.intercept_

#BONUS: modelo com SM para fazer a BackWard Elimination
X = X[:, 1:]
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
sm_regressor = sm.OLS(y, X_opt).fit()
sm_regressor.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
sm_regressor = sm.OLS(y, X_opt).fit()
sm_regressor.summary()
X_opt = X[:, [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
sm_regressor = sm.OLS(y, X_opt).fit()
sm_regressor.summary()
X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
sm_regressor = sm.OLS(y, X_opt).fit()
sm_regressor.summary()

#