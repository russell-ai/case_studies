"""Simple Linear Regression"""

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% Importing the dataset
df = pd.read_csv("datasets/Salary.csv")

#%% check if there are any Null values
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()
#%% Check the dataframe info

df.info()

#%% Statistical summary of the dataframe
df.describe()

#%%
df.hist(bins = 30, figsize = (20,10), color = 'r')
plt.show()

#%% plot pairplot
sns.pairplot(df)
plt.show()

#%%
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()

#%% md  regplot in Seaborn to obtain a straight line fit between "salary" and "years of experience"
sns.regplot(x="YearsExperience", y="Salary", data=df)
plt.show()

#%% Dependent and Independent variables
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Check the shape

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=4)



#%% Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


#%% Predicting the Test set results
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
y_pred = lr.predict(X_test)

regresssion_model_accuracy = lr.score(X_test,y_test)
print(regresssion_model_accuracy)

#%% Compare the results
result = pd.DataFrame({"True Value":y_test,"Predict Value":y_pred})

#%% Visualising the Model results

plt.scatter(X_train,y_train, color="blue")
plt.scatter(X_test,y_test, color="black")
plt.plot(X_train,lr.predict(X_train), color="red")
plt.title("Salary vs. Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()


#%% Making a single prediction
# Getting the final linear regression equation with the values of the coefficients (y=mx+b)
print('Linear Model Coefficient (m): ', lr.coef_)
print('Linear Model Coefficient (b): ', lr.intercept_)


print("Salary for 11 years experience:")

print(lr.predict([[11]]))