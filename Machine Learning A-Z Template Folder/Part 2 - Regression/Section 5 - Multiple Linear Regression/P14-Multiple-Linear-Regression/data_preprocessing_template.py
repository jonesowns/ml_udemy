# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values



#   Encoding the catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#one hot encoder is depracted, using col transformer instead
#labelencoder_X = LabelEncoder()
#X[:, 0] =  labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer([("one_hot_encoder", OneHotEncoder(), [3])], remainder='passthrough')

#learn from this mistake, could not display the varaible X as array.
#X = ct.fit_transform(X).toarray()[:4]

X = np.array(ct.fit_transform(X), dtype=np.float)


#Avoiding the dummy variable trap

X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

#sklearn.cross_validation is depracated. use below.... model_selection

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


 



#fitting multiple linear regressot to the trianing set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(X_test)



#Visualizong the Training set results


'''plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs experience (Training Set)' )
plt.xlabel('x label')
plt.ylabel('y label')
plt.show'''


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#Building the optimal model using backward elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X , axis = 1 )
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#X_opt = X[:,[0,1,2,3,4,5]]
# Start with all independant variables, then keep removing variables with hig P value



X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
