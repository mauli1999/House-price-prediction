import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import keras

# Importing the dataset
dataset = pd.read_csv('data_hackathon_pdpu.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 18].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 17] = labelencoder_X.fit_transform(X[:, 17])
onehotencoder = OneHotEncoder(categorical_features = [0,2,17])

X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.05)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X, y)

pickle.dump(regressor,open('RFR.pkl','wb'))
model=pickle.load(open('RFR.pkl','rb'))
regressor.decision_path
# Predicting a new result
y_pred = regressor.predict(X_test)

print(regressor.score(X_test,Y_test))

lengths = [x for x in range(len(X_test))]

plt.scatter(lengths,Y_test,edgecolors= "green")
plt.scatter(lengths,y_pred,edgecolors = "red")
plt.show()