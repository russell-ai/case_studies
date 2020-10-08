"""Data Preprocessing"""

#%% Importing the libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml

#%% Importing the data
x,y = fetch_openml("titanic",version=1, as_frame=True, return_X_y=True)

#%% Reviews of data - first insight
x.head(3)

#%% Column Tranformation - Select cat and Continuous Features

cat_var = ['embarked', 'sex', 'pclass']
cont_var = ['age', 'fare']

#%% Create a pipeline of both Continuous and Categorical Variables

cont_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale",StandardScaler())])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

#%% Apply Column Transformers
# ColumnTransformer [('name', pipeline_name, features)]

preprocess_pipeline = ColumnTransformer([
    ("continues", cont_pipeline, cont_var),
    ("categorical", cat_pipeline, cat_var)
])


#%% Apply Fit_transform(Data)

xtrain = preprocess_pipeline.fit_transform(x)
