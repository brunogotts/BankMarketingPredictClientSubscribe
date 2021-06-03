# data analysis and wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# encoding
#from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("bank-full.csv",sep=';')

print(dataset.columns.values)

dataset.head()

# removing columns that won't be used for prediction
columns_removed = ['day','month'] 
dataset = dataset.drop(columns_removed, axis = 1)

# checking null values
print(dataset.columns.values)
dataset.isna().sum()

# encoding boolean 
dataset = dataset.replace({'no': 0, 'yes': 1})
#dataset.dtypes

# One Hot Encoding for categorical variables
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = dataset.index

# Remove categorical columns (will replace with one-hot encoding)
num_train = dataset.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
dataset = pd.concat([num_train, OH_cols], axis=1)

# Checking dataset
dataset.head()

X = dataset.loc[:, dataset.columns != 'y'].values
y = dataset.loc[:, dataset.columns == 'y'].values

# Spliting train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Setting standard scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Predicting with some methods 
# 1) LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train.ravel())

# Predicting results with y_pred
y_pred = classifier.predict(X_test)

# Validating results with Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# 2) Forest Tree
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train, y_train.ravel())
# Predicting results with y_pred
y_pred = classifier.predict(X_test)

# Validating results with Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# 3) Random Forest
classifier = RandomForestClassifier(random_state = 0)
classifier.fit(X_train, y_train.ravel())
# Predicting results with y_pred
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)

# 4) GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train.ravel())
# Predicting results with y_pred
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)

# 5) KNeighbors
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train.ravel())
# Predicting results with y_pred
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)

# Winner: KNeighbors