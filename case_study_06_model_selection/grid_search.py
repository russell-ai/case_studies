"""Grid Search"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('case_study_06_model_selection\Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 4)
classifier.fit(X_train, y_train)

#%% Making Prediction and the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = accuracy_score(y_test, y_pred)
print(score)

#%% Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(f"Accuracy :{accuracies.mean()*100:.2f}")
print(f"Standart Deviation: {accuracies.std()*100:.2f}")


#%% Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{"C":[0.5,0.75,1],"kernel":["linear"]},
              {"C":[0.5,0.75,1],"kernel":["rbf"],"gamma":[0.6,0.8,1]},
              {"C":[0.5,0.75,1],"kernel":["poly"],"degree":[2,3],"gamma":[0.6,0.8,1]},
              {"C":[0.5,0.75,1],"kernel":["sigmoid"],"gamma":[0.6,0.8,1]}]
gird_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=10,
                           n_jobs=-1)
gird_search.fit(X_train,y_train)
print("Best Score:",gird_search.best_score_)
print("Best Parameters:",gird_search.best_params_)

