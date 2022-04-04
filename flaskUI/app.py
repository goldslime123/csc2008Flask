from flask import Flask, render_template
import os

# postgres
import psycopg2

# MariaDB Imports
import mariadb
import sys

#linear regression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__, template_folder="Website")
IS_DEV = app.env == 'development'


@app.route('/')
def index():

    list = []
    conn = None

    # Connect to postgresql Platform
    try:
        conn = psycopg2.connect(
            host="ec2-54-173-77-184.compute-1.amazonaws.com",
            database="d2v75ijfptfl5f",
            user="jkbetvbzvsivpk",
            password=
            "3b79c1f6062e3164cb523ea49ade123ccc4d25a86f7fa9c7e2b42921d0f55831")

        print("Successfully connected", file=sys.stderr)

    except Exception as error:
        print("Error connecting to Postgres Platform: {}".format(error))

    # Get Cursor
    if conn is not None:
        # linear regression training
        LR_temperature()

        # show linear and non linear
        cur = conn.cursor()
        cur.execute(
            "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;"
        )

        for i in cur:
            list.append(i)

        data = [(item[0], float(item[1]), float(item[2]), float(item[3]),
                 float(item[4])) for item in list]
        for i in data:
            print(i, file=sys.stderr)

        #from database
        labels = [row[0] for row in data]
        labels.append('2022.1')

        temperature = [row[1] for row in data]

        electricPrice = [row[2] for row in data]
        crudePrice = [row[3] for row in data]
        maintenance = [row[4] for row in data]
        return render_template("index.html",
                               labels=labels,
                               electricPrice=electricPrice,
                               crudePrice=crudePrice,
                               temperature=temperature,
                               maintenance=maintenance)

        

    # # This is where we import our data from database
    # data = [
    #     ("2015-Q1", 23.29, 25.6, 27.4, 21.9),
    #     ("2015-Q2", 20.87, 25.6, 28.8, 21.9),
    #     ("2015-Q3", 22.41, 25.6, 28.7, 21.9),
    #     ("2015-Q4", 20.35, 27.9, 28.1, 21.9),
    #     ("2016-Q1", 19.5, 27.9, 28.4, 24.4),
    #     ("2016-Q2", 17.68, 27.9, 29.1, 24.4),
    #     ("2016-Q3", 19.28, 27.9, 27.7, 24.4),
    #     ("2016-Q4", 19.13, 27.9, 27.1, 24.4),
    #     ("2017-Q1", 20.2, 24.2, 28.3, 24.9),
    # ]

    # labels = [row[0] for row in data]
    # electricPrice = [row[1] for row in data]
    # crudePrice = [row[2] for row in data]
    # temperature = [row[3] for row in data]
    # maintenance = [row[4] for row in data]
    # return render_template("index.html", labels=labels, electricPrice=electricPrice, crudePrice=crudePrice, temperature=temperature, maintenance=maintenance )



#Predict tempeature
predictedTemp = 0
predictedTarriff_temp =0
def LR_temperature():
    # import data set
    datasetTemp = pd.read_csv('csv/weather.csv')
    XTemp = datasetTemp['quarter'].values.reshape(-1, 1)
    yTemp = datasetTemp['temperature'].values.reshape(-1, 1)

    datasetTariff = pd.read_csv('csv/tariff.csv')
    XTariff = datasetTemp['temperature'].values.reshape(-1, 1)
    yTariff = datasetTariff['tariff_per_kwh'].values.reshape(-1, 1)

    # split data set to training/test set 80% traning
    X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = train_test_split(
        XTemp, yTemp, test_size=0.2, random_state=0)
    X_trainTariff, X_testTariff, y_trainTariff, y_testTariff = train_test_split(
        XTariff, yTariff, test_size=0.2, random_state=0)

    # train training set
    regressorTemp = LinearRegression()
    regressorTemp.fit(X_trainTemp, y_trainTemp)

    regressorTariff = LinearRegression()
    regressorTariff.fit(X_trainTariff, y_trainTariff)

    #Predict Temperature
    # test data and see how accurately our algorithm predicts the percentage score.
    y_predTemp = regressorTemp.predict(X_testTemp)
    print('Mean Absolute Error:',
            metrics.mean_absolute_error(y_testTemp, y_predTemp))
    print('Mean Squared Error:',
            metrics.mean_squared_error(y_testTemp, y_predTemp))
    print('Root Mean Squared Error:',
            np.sqrt(metrics.mean_squared_error(y_testTemp, y_predTemp)))
    # get intercept:
    print("y intercept: " + str(regressorTemp.intercept_))
    # get slope:
    print("slope: " + str(regressorTemp.coef_))
    # y = a+bx
    global predictedTemp
    predictedTemp = regressorTemp.intercept_ + (regressorTemp.coef_ * 2022.1)
    print("Predicted Temperature: 2022 Quarter 1: " + str(predictedTemp))

    # show training set
    plt.scatter(X_trainTemp, y_trainTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Temperature (Training set)')
    plt.xlabel('Quarter')
    plt.ylabel('Temperature')
    plt.savefig('image/linear/temperature/temp_quarter_train.png')
    plt.close()

    #show test set
    plt.scatter(X_testTemp, y_testTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Temperature (Test set)')
    plt.xlabel('Quarter')
    plt.ylabel('Temperature')
    plt.savefig('image/linear/temperature/temp_quarter_test.png')
    plt.close()

    #Predict Tariff
    y_predTariff = regressorTariff.predict(X_testTariff)
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(y_testTariff, y_predTariff))
    print('Mean Squared Error:',
          metrics.mean_squared_error(y_testTariff, y_predTariff))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_testTariff, y_predTariff)))

    print("y intercept: " + str(regressorTariff.intercept_))
    print("slope: " + str(regressorTariff.coef_)) 

    global predictedTarriff_temp
    predictedTarriff_temp = regressorTariff.intercept_ + (regressorTariff.coef_ *
                                                    predictedTemp)
    print("Predicted Tariff: 2022 Quarter 1: " + str(predictedTarriff_temp))  

    
    plt.scatter(X_trainTariff, y_trainTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Temperature vs Tariff (Training set)')
    plt.xlabel('Temperature')
    plt.ylabel('Tariff')
    plt.savefig('image/linear/temperature/temp_tariff_train.png')
    plt.close()

    plt.scatter(X_testTariff, y_testTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Temperature vs Tariff (Test set)')
    plt.xlabel('Temperature')
    plt.ylabel('Tariff')
    plt.savefig('image/linear/temperature/temp_tariff_test.png')
    plt.close()


if __name__ == 'main':
    # guaranteed to run on production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
