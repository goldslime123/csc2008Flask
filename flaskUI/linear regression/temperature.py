import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


datasetTemp = pd.read_csv('../csv/weather.csv')
XTemp = datasetTemp['quarter'].values.reshape(-1, 1)
yTemp = datasetTemp['temperature'].values.reshape(-1, 1)

# split data set to training/test set 80% traning
X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = train_test_split(
      XTemp, yTemp, test_size=0.2, random_state=0)

# train training set
regressorTemp = LinearRegression()
regressorTemp.fit(X_trainTemp, y_trainTemp)


#Predict tempeature
# get intercept:
print("y intercept: " + str(regressorTemp.intercept_))
# get slope:
print("slope: " + str(regressorTemp.coef_))

# y = a+bx
# test data and see how accurately our algorithm predicts the percentage score.
y_predTemp = regressorTemp.predict(X_testTemp)

# mae, msq, rmse
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_testTemp, y_predTemp))
print('Mean Squared Error:',
      metrics.mean_squared_error(y_testTemp, y_predTemp))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_testTemp, y_predTemp)))

predictedTemp = regressorTemp.intercept_ + (regressorTemp.coef_ * 2022.1)
print("Predicted Temperature: 2022 Quarter 1: " + str(predictedTemp))

# show training set
plt.scatter(X_trainTemp, y_trainTemp, color='red')
plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
plt.title('Quarter vs Temperature (Training set)')
plt.xlabel('Quarter')
plt.ylabel('Temperature')
plt.show()
#show test set
plt.scatter(X_testTemp, y_testTemp, color='red')
plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
plt.title('Quarter vs Temperature (Test set)')
plt.xlabel('Quarter')
plt.ylabel('Temperature')
plt.show()

# Predict tariff
datasetTariff = pd.read_csv('../csv/tariff.csv')
XTariff = datasetTemp['temperature'].values.reshape(-1, 1)
yTariff  = datasetTariff['tariff_per_kwh'].values.reshape(-1, 1)

X_trainTariff, X_testTariff, y_trainTariff, y_testTariff = train_test_split(XTariff, yTariff, test_size = 0.2, random_state = 0)

regressorTariff = LinearRegression()
regressorTariff.fit(X_trainTariff, y_trainTariff)

print("y intercept: " + str(regressorTariff.intercept_))
print("slope: " + str(regressorTariff.coef_))

y_predTariff = regressorTariff.predict(X_testTariff)

print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_testTariff, y_predTariff))
print('Mean Squared Error:',
      metrics.mean_squared_error(y_testTariff, y_predTariff))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_testTariff, y_predTariff)))

predictedTariff = regressorTariff.intercept_ + (regressorTariff.coef_ * predictedTemp)
print("Predicted Tariff: 2022 Quarter 1: " + str(predictedTariff))

plt.scatter(X_trainTariff, y_trainTariff, color='red')
plt.plot(X_trainTariff,
            regressorTariff.predict(X_trainTariff),
            color='blue')
plt.title('Temperature vs Tariff (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Tariff')
plt.show()
plt.scatter(X_testTariff, y_testTariff, color='red')
plt.plot(X_trainTariff,
            regressorTariff.predict(X_trainTariff),
            color='blue')
plt.title('Temperature vs Tariff (Testing set)')
plt.xlabel('Temperature')
plt.ylabel('Tariff')
plt.show()