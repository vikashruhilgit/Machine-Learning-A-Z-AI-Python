# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
dataset = pd.read_csv('iris.csv')
# Separate features and target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split the dataset into an 80-20 training-test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# Apply feature scaling on the training and test sets

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Print the scaled training and test sets
print(X_train)
print(X_test)
