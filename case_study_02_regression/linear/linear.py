"""Simple Linear Regression"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
df = pd.read_csv("Salary_Data.csv")
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=4)



#%% Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


#%% Predicting the Test set results
y_pred = lr.predict(X_test)

lr.score(X_test,y_test)

result = pd.DataFrame({"True Value":y_test,"Predict Value":y_pred})

#%% Visualising the Training set results

plt.scatter(X_train,y_train)
plt.plot(X_train,lr.predict(X_train), color="red")
plt.title("Training Set - Experience & Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()

#%% Visualising the Test set results

plt.scatter(X_test,y_test)
plt.plot(X_train,lr.predict(X_train), color="red")
# plt.scatter(X_test,lr.predict(X_test), color="yellow")
plt.title("Test Set - Experience & Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()

#%% Making a single prediction
# Getting the final linear regression equation with the values of the coefficients (y=ax+b)
print(lr.coef_)
print(lr.intercept_)

print("Salary for 11 years experience:")

print(lr.predict([[11]]))