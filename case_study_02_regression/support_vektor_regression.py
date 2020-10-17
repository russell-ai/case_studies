"""Support Vector Regression (SVR)"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
df = pd.read_csv("datasets/Position_Salaries.csv")
X = df.iloc[:,1:-1].values   # ndim is 2 array
y = df.iloc[:,-1].values     # ndim is 1 array
y = y.reshape(len(y),1)      # make 2 array. Now check shape and ndim.

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()     # this scaler is for X data
sc_y = StandardScaler()     # this scaler  is for y data
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)   # be careful this y label is not suitable for label encoder or one hot encoder.


#%% Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X,y)

#%% Predicting a new result
test_data = [[6.5]]
scaled_test_data = sc_X.transform(test_data)
prediction = regressor.predict(scaled_test_data)
prediction = sc_y.inverse_transform((prediction))
print(prediction)

#%% Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#%% Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()