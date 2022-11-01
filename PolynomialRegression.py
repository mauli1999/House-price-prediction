import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data_hackathon_pdpu.csv")
dataset = dataset.drop('Area',axis=1)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,17].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 16] = labelencoder_X.fit_transform(X[:, 16])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_poly,y_train)

y_pred = linear_regressor.predict(X_test_poly)
X_test = sc_X.inverse_transform(X_test)
for i in range(len(X_test)):
    print(X_test[i], " : ", y_pred[i])

print("Accuracy = ",linear_regressor.score(X_test_poly,y_test))

