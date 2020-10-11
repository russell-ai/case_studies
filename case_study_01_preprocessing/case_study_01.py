"""Data Preprocessing"""

#%% Importing the libraries (numpy,pandas,matplotlib,...)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

#%% Importing the data
# df = pd.read_csv("datasets/Data.csv")
#Old custom way - reading file
# import csv
# data = open("Data.csv","r")
# file= csv.reader(data)
# for row in file:
#     print(row)

#%% Reviews of data - first insight
# df.head(3)
# print(df.size, df.shape)
# df.describe().T
# df.info()
# df.isnull()
# df.isnull().sum()
# sns.heatmap(df.isnull())
# plt.show()
#%% Taking care of missing data (fill nan values with consistent value)

# df.dropna(inplace=True)

# df.fillna(method="pad", inplace=True)

# df.interpolate(method="linear")

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# imputer.fit(df.iloc[:,1:3])
# df.iloc[:,1:3] = imputer.transform(df.iloc[:,1:3])

# df.groupby("Country").mean()
# df["Age"] = df.groupby("Country")["Age"].apply(lambda x:x.fillna(x.mean()))
# df["Salary"] = df.groupby("Country")["Salary"].apply(lambda x:x.fillna(x.mean()))

#%% Saving the clean data
# df.to_csv("CleanData.csv")

# #%% Encoding the categorical data - Independent Variable
# x = df.iloc[:,1:-1]
# y = df.iloc[:,-1]

# df_encoded = pd.get_dummies(df,columns=["Country"], drop_first=True)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = np.array(ct.fit_transform(x))




#%% Encoding the categorical data - Dependent Variable

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y= le.fit_transform(y)

#%% Splitting the dataset into the Training set and Test set

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y)

#%%Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(x_train)
# x_train = sc.transform(x_train)
# x_test = sc.transform(x_test)