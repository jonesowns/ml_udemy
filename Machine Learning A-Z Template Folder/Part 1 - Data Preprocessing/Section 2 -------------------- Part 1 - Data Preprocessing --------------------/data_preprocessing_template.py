# Data Preprocessing


# Importing libraires`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Data.csv')


# make the matrix of features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean' )
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#   Encoding the catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#one hot encoder is depracted, using col transformer instead
#labelencoder_X = LabelEncoder()
#X[:, 0] =  labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer([("one_hot_encoder", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

labelencoder_y = LabelEncoder()
y =  labelencoder_y.fit_transform(y)



# Split the dataset into Training and Test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , train_size = 0.8, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
 