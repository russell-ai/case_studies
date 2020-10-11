"""Logistic Regression"""

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% importing the dataset
df = pd.read_csv("datasets/breast_cancer.csv")

#%%
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#%% Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#%% Predicting the Test set results
y_pred = classifier.predict((X_test))

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)
print("Confusion Matrix:\n",cm)
print("Accuracy Score:",score)



#%% Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f}".format(accuracies.mean()))
print("Standart Deviation :{:.2f}".format(accuracies.std()))

