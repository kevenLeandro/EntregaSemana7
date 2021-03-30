import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

dataset = pd.read_csv('winequality-red.csv')

ind = dataset.iloc[:, [7,10]].values
dep = dataset.iloc[:, -1].values

# arvore Aleatoria
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ind,dep, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
randomForestRegressor = RandomForestRegressor (n_estimators=10,random_state=0)
randomForestRegressor.fit(ind,dep)

Y_pred = randomForestRegressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred) )

# regressao multipla
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ind,dep, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))

# regressao polimonial

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ind,dep, test_size=0.2, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)
regressor = LinearRegression ()
regressor.fit(X_poly, Y_train)
Y_pred = regressor.predict(poly.transform(X_test))

from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred)) # baixar o grau pode aumentar  o r2_score


