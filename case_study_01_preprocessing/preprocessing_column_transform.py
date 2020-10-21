"""Data Preprocessing Column Transformation"""

#%% Importing the libraries

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

#%% Importing the data
X,y = fetch_openml("titanic",version=1, as_frame=True, return_X_y=True)

#%% Column Tranformation - Select cat and Continuous Features

cat_var = ['embarked', 'sex', 'pclass']
cont_var = ['age', 'fare']

X = X[cat_var + cont_var].sample(10)


#%% Column Transformation with Sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder(handle_unknown='ignore')
imp = SimpleImputer()

ct = make_column_transformer(
    (ohe, cat_var),
    (imp, cont_var),
    remainder="passthrough"
)

X_train = ct.fit_transform(X)
