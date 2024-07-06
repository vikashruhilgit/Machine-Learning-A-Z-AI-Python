# Importing the necessary libraries
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')


# Identify missing data (assumes that missing data is represented as NaN)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Print the number of missing entries in each column
print(x)
print(y)
# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit the imputer on the DataFrame
imputer.fit(x[:, :])
# Apply the transform to the DataFrame
x[:, :] = imputer.transform(x[:, :])
#Print your updated matrix of features
print(x)