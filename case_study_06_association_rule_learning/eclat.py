"""Eclat"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Data Preprocessing
df = pd.read_csv('datasets/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, df.shape[0]):
  transactions.append([str(df.values[i,j]) for j in range(0, df.shape[1])])

#%% Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#%% Visualising the results
## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

#%% Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]         # we removed confidence and lift that the apriori model has.
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

#%% Displaying the results sorted by descending supports
resultsinDataFrame.nlargest(n = 10, columns = 'Support')