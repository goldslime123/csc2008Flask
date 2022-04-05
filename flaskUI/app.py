from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template
import os

# postgres
import psycopg2

# MariaDB Imports
import mariadb
import sys

# linear regression
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, template_folder="Website")
IS_DEV = app.env == 'development'

# Images
imageFolder = os.path.join('static', 'image')
app.config['UPLOAD_FOLDER'] = imageFolder

#year
years = [2022.1, 2022.2, 2022.3, 2022.4, 2023.1, 2023.2, 2023.3, 2023.4]

# Predict tariff based on temperature
linearResultPredictedTemp = []
linearResultPredictedTarriffTemp = []

# Predict tariff based on maintenance
linearResultPredictedMain = []
linearResultPredictedTarriffMain = []

# Predict tariff based on crudeoil
linearResultPredictedCrude = []
linearResultPredictedTarriffCrude = []


#################################################################################
############################ HomePage ###########################################
#################################################################################
@app.route('/')
def home():

    #load gif image:
    gifImage = os.path.join(app.config['UPLOAD_FOLDER'], 'LightBulb.gif')

    return render_template("home.html", gifImage=gifImage)


#################################################################################
############################ Temperature Page ###################################
#################################################################################
@app.route('/temperature')
def temperature():
    list = []
    conn = None

    # linear regression training
    lr_temperature()
    # load images
    lr_temp_quarter_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                         'lr_temp_quarter_train.png')
    lr_temp_quarter_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                        'lr_temp_quarter_test.png')
    lr_temp_tariff_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                        'lr_temp_tariff_train.png')
    lr_temp_tariff_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                       'lr_temp_tariff_test.png')

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

        # show linear and non linear
        cur = conn.cursor()
        cur.execute(
            "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;"
        )

        for i in cur:
            list.append(i)

        conn.close()

        linearData = [(item[0], float(item[1]), float(item[2]))
                      for item in list]

        nonLinearData = [(item[0], float(item[1]), float(item[2]))
                         for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear
        l_labels = [row[0] for row in linearData]
        # add years
        for x in years:
            l_labels.append(x)

        l_temperature = [row[1] for row in linearData]
        # add predicted temp
        temp = []
        for x in linearResultPredictedTemp:
            temp.append(np.round(x, 2))
        for x in temp:
            l_temperature.append(x)

        l_electricPrice = [row[2] for row in linearData]
        # add predicted tariff
        temp1 = []
        for x in linearResultPredictedTarriffTemp:
            temp1.append(np.round(x, 2))
        for x in temp1:
            l_electricPrice.append(x)

        # Non Linear
        nl_labels = [row[0] for row in nonLinearData]
        nl_labels.append('2022.1')
        nl_temperature = [row[1] for row in nonLinearData]
        nl_electricPrice = [row[2] for row in nonLinearData]

        return render_template(
            "temperature.html",
            l_labels=l_labels,
            l_electricPrice=l_electricPrice,
            l_temperature=l_temperature,
            nl_labels=nl_labels,
            nl_electricPrice=nl_electricPrice,
            nl_temperature=nl_temperature,

            # images
            lr_temp_quarter_train=lr_temp_quarter_train,
            lr_temp_quarter_test=lr_temp_quarter_test,
            lr_temp_tariff_train=lr_temp_tariff_train,
            lr_temp_tariff_test=lr_temp_tariff_test,
        )


#################################################################################
############################ Maintenance Page ###################################
#################################################################################
@app.route('/maintenance')
def maintenance():

    list = []
    conn = None

    # linear regression training
    lr_maintenance()

    # load images
    lr_main_quarter_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                         'lr_main_quarter_train.png')
    lr_main_quarter_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                        'lr_main_quarter_test.png')
    lr_main_tariff_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                        'lr_main_tariff_train.png')
    lr_main_tariff_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                       'lr_main_tariff_test.png')

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

        # show linear and non linear
        cur = conn.cursor()
        cur.execute(
            "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;"
        )

        for i in cur:
            list.append(i)

        conn.close()

        linearData = [(item[0], float(item[2]), float(item[4]))
                      for item in list]

        nonLinearData = [(item[0], float(item[2]), float(item[4]))
                         for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear
        l_labels = [row[0] for row in linearData]
        # add years
        for x in years:
            l_labels.append(x)

        l_maintenance = [row[1] for row in linearData]
        # add predicted maintenance
        temp = []
        for x in linearResultPredictedMain:
            temp.append(np.round(x, 2))
        for x in temp:
            l_maintenance.append(x)

        l_electricPrice = [row[2] for row in linearData]
        # add predicted tariff
        temp1 = []
        for x in linearResultPredictedTarriffMain:
            temp1.append(np.round(x, 2))
        for x in temp1:
            l_electricPrice.append(x)

        # Non Linear
        nl_labels = [row[0] for row in nonLinearData]
        nl_labels.append('2022.1')
        nl_electricPrice = [row[1] for row in nonLinearData]
        nl_maintenance = [row[2] for row in nonLinearData]

        return render_template(
            "maintenance.html",
            l_labels=l_labels,
            l_electricPrice=l_electricPrice,
            l_maintenance=l_maintenance,
            nl_labels=nl_labels,
            nl_electricPrice=nl_electricPrice,
            nl_maintenance=nl_maintenance,
            #images
            lr_main_quarter_train=lr_main_quarter_train,
            lr_main_quarter_test=lr_main_quarter_test,
            lr_main_tariff_train=lr_main_tariff_train,
            lr_main_tariff_test=lr_main_tariff_test,
        )


#################################################################################
############################ Crude Oil Page #####################################
#################################################################################
@app.route('/crudeoil')
def crudeoil():

    list = []
    conn = None

    # linear regression training
    lr_crudeoil()
    # load images
    lr_crudeoil_quarter_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                             'lr_crudeoil_quarter_train.png')
    lr_crudeoil_quarter_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                            'lr_crudeoil_quarter_test.png')
    lr_crudeoil_tariff_train = os.path.join(app.config['UPLOAD_FOLDER'],
                                            'lr_crudeoil_tariff_train.png')
    lr_crudeoil_tariff_test = os.path.join(app.config['UPLOAD_FOLDER'],
                                           'lr_crudeoil_tariff_test.png')

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

        # show linear and non linear
        cur = conn.cursor()
        cur.execute(
            "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;"
        )

        for i in cur:
            list.append(i)

        conn.close()

        linearData = [(item[0], float(item[2]), float(item[3]))
                      for item in list]

        nonLinearData = [(item[0], float(item[2]), float(item[3]))
                         for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear

        # Linear
        l_labels = [row[0] for row in linearData]
        # add years
        for x in years:
            l_labels.append(x)

        l_crudePrice = [row[1] for row in linearData]
        # add predicted crude oil
        temp = []
        for x in linearResultPredictedCrude:
            temp.append(np.round(x, 2))
        for x in temp:
            l_crudePrice.append(x)

        l_electricPrice = [row[2] for row in linearData]
        # add predicted tariff
        temp1 = []
        for x in linearResultPredictedTarriffCrude:
            temp1.append(np.round(x, 2))
        for x in temp1:
            l_electricPrice.append(x)

        # Non Linear
        nl_labels = [row[0] for row in nonLinearData]
        nl_labels.append('2022.1')
        nl_electricPrice = [row[1] for row in nonLinearData]
        nl_crudePrice = [row[2] for row in nonLinearData]

        return render_template(
            "crudeoil.html",
            l_labels=l_labels,
            l_electricPrice=l_electricPrice,
            l_crudePrice=l_crudePrice,
            nl_labels=nl_labels,
            nl_electricPrice=nl_electricPrice,
            nl_crudePrice=nl_crudePrice,

            # images
            lr_crudeoil_quarter_train=lr_crudeoil_quarter_train,
            lr_crudeoil_quarter_test=lr_crudeoil_quarter_test,
            lr_crudeoil_tariff_train=lr_crudeoil_tariff_train,
            lr_crudeoil_tariff_test=lr_crudeoil_tariff_test,
        )


#################################################################################
############################ Linear ML Functions ################################
#################################################################################
def lr_temperature():
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

    data = []
    cur = conn.cursor()
    cur.execute(
        "SELECT w.quarter, w.temperature, t.tariff_per_kwh FROM weather w, tariff t WHERE w.quarter=t.quarter;"
    )

    for i in cur:
        data.append(i)

    conn.close()

    df = pd.DataFrame(data)

    # saving the dataframe
    df.to_csv('temp/temperature.csv',
              header=['quarter', 'temperature', 'tariff_per_kwh'],
              index=False)

    # import data set
    dataset = pd.read_csv('temp/temperature.csv')

    XTemp = dataset['quarter'].values.reshape(-1, 1)
    yTemp = dataset['temperature'].values.reshape(-1, 1)

    XTariff = dataset['temperature'].values.reshape(-1, 1)
    yTariff = dataset['tariff_per_kwh'].values.reshape(-1, 1)

    # split data set to training/test set 80% traning
    X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = train_test_split(
        XTemp, yTemp, test_size=0.4, random_state=0)
    X_trainTariff, X_testTariff, y_trainTariff, y_testTariff = train_test_split(
        XTariff, yTariff, test_size=0.4, random_state=0)

    # train training set
    regressorTemp = LinearRegression()
    regressorTemp.fit(X_trainTemp, y_trainTemp)

    regressorTariff = LinearRegression()
    regressorTariff.fit(X_trainTariff, y_trainTariff)

    # Predict Temperature
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
    global linearResultPredictedTemp

    # for x in years:
    #     predictedTemp = regressorTemp.intercept_ + (regressorTemp.coef_ * x)
    #     predictedTemp = np.round(predictedTemp,2)
    #     linearResultPredictedTemp.append(predictedTemp)

    linearResultPredictedTemp = y_predTemp.flatten()
    print(linearResultPredictedTemp)

    # show training set
    plt.scatter(X_trainTemp, y_trainTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Temperature (Training set)')
    plt.xlabel('Quarter')
    plt.ylabel('Temperature')
    plt.savefig('../flaskUI/static/image/lr_temp_quarter_train.png')
    plt.close()

    # show test set
    plt.scatter(X_testTemp, y_testTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Temperature (Test set)')
    plt.xlabel('Quarter')
    plt.ylabel('Temperature')
    plt.savefig('../flaskUI/static/image/lr_temp_quarter_test.png')
    plt.close()

    # Predict Tariff
    y_predTariff = regressorTariff.predict(X_testTariff)
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(y_testTariff, y_predTariff))
    print('Mean Squared Error:',
          metrics.mean_squared_error(y_testTariff, y_predTariff))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_testTariff, y_predTariff)))

    print("y intercept: " + str(regressorTariff.intercept_))
    print("slope: " + str(regressorTariff.coef_))

    global linearResultPredictedTarriffTemp
    a = str(regressorTariff.intercept_)
    intecept = str(a).replace('[',
                              '').replace(']',
                                          '').replace('\'',
                                                      '').replace('\"', '')
    b = str(regressorTariff.coef_)
    slope = str(b).replace('[', '').replace(']',
                                            '').replace('\'',
                                                        '').replace('\"', '')
    for x in linearResultPredictedTemp:
        predictedTariffTemp = float(intecept) + (float(slope) * x)
        predictedTariffTemp = np.round(predictedTariffTemp, 2)
        linearResultPredictedTarriffTemp.append(predictedTariffTemp)

    plt.scatter(X_trainTariff, y_trainTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Temperature vs Tariff (Training set)')
    plt.xlabel('Temperature')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_temp_tariff_train.png')
    plt.close()

    plt.scatter(X_testTariff, y_testTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Temperature vs Tariff (Test set)')
    plt.xlabel('Temperature')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_temp_tariff_test.png')
    plt.close()


def lr_maintenance():
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

    data = []
    cur = conn.cursor()
    cur.execute(
        "SELECT m.quarter, m.cost, t.tariff_per_kwh FROM maintenance m, tariff t WHERE m.quarter=t.quarter;"
    )

    for i in cur:
        data.append(i)

    conn.close()

    df = pd.DataFrame(data)

    # saving the dataframe
    df.to_csv('temp/maintenance.csv',
              header=['quarter', 'maintenance', 'tariff_per_kwh'],
              index=False)

    # import data set
    dataset = pd.read_csv('temp/maintenance.csv')

    XTemp = dataset['quarter'].values.reshape(-1, 1)
    yTemp = dataset['maintenance'].values.reshape(-1, 1)

    XTariff = dataset['maintenance'].values.reshape(-1, 1)
    yTariff = dataset['tariff_per_kwh'].values.reshape(-1, 1)

    # split data set to training/test set 80% traning
    X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = train_test_split(
        XTemp, yTemp, test_size=0.4, random_state=0)
    X_trainTariff, X_testTariff, y_trainTariff, y_testTariff = train_test_split(
        XTariff, yTariff, test_size=0.4, random_state=0)

    # train training set
    regressorTemp = LinearRegression()
    regressorTemp.fit(X_trainTemp, y_trainTemp)

    regressorTariff = LinearRegression()
    regressorTariff.fit(X_trainTariff, y_trainTariff)

    # Predict Temperature
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
    global linearResultPredictedMain

    linearResultPredictedMain = y_predTemp.flatten()

    # show training set
    plt.scatter(X_trainTemp, y_trainTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Maintenance (Training set)')
    plt.xlabel('Quarter')
    plt.ylabel('Maintenance')
    plt.savefig('../flaskUI/static/image/lr_main_quarter_train.png')
    plt.close()

    # show test set
    plt.scatter(X_testTemp, y_testTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Maintenance (Test set)')
    plt.xlabel('Quarter')
    plt.ylabel('Maintenance')
    plt.savefig('../flaskUI/static/image/lr_main_quarter_test.png')
    plt.close()

    # Predict Tariff
    y_predTariff = regressorTariff.predict(X_testTariff)
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(y_testTariff, y_predTariff))
    print('Mean Squared Error:',
          metrics.mean_squared_error(y_testTariff, y_predTariff))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_testTariff, y_predTariff)))

    print("y intercept: " + str(regressorTariff.intercept_))
    print("slope: " + str(regressorTariff.coef_))

    global linearResultPredictedTarriffMain
    # linearResultPredictedTarriffMain = y_predTariff.flatten()

    a = str(regressorTariff.intercept_)
    intecept = str(a).replace('[',
                              '').replace(']',
                                          '').replace('\'',
                                                      '').replace('\"', '')
    print(intecept)
    b = str(regressorTariff.coef_)
    slope = str(b).replace('[', '').replace(']',
                                            '').replace('\'',
                                                        '').replace('\"', '')
    print(slope)

    for x in linearResultPredictedMain:
        predictedTariffMain = float(intecept) + (float(slope) * x)
        predictedTariffMain = np.round(predictedTariffMain, 2)
        linearResultPredictedTarriffMain.append(predictedTariffMain)

    plt.scatter(X_trainTariff, y_trainTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Maintenance vs Tariff (Training set)')
    plt.xlabel('Maintenance')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_main_tariff_train.png')
    plt.close()

    plt.scatter(X_testTariff, y_testTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Maintenance vs Tariff (Test set)')
    plt.xlabel('Maintenance')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_main_tariff_test.png')
    plt.close()


def lr_crudeoil():
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

    data = []
    cur = conn.cursor()
    cur.execute(
        "SELECT c.quarter, c.price_per_barrel, t.tariff_per_kwh FROM crudeoil c, tariff t WHERE c.quarter=t.quarter;"
    )

    for i in cur:
        data.append(i)

    conn.close()

    df = pd.DataFrame(data)

    # saving the dataframe
    df.to_csv('temp/crudeoil.csv',
              header=['quarter', 'price_per_barrel', 'tariff_per_kwh'],
              index=False)

    # import data set
    dataset = pd.read_csv('temp/crudeoil.csv')

    XTemp = dataset['quarter'].values.reshape(-1, 1)
    yTemp = dataset['price_per_barrel'].values.reshape(-1, 1)

    XTariff = dataset['price_per_barrel'].values.reshape(-1, 1)
    yTariff = dataset['tariff_per_kwh'].values.reshape(-1, 1)

    # split data set to training/test set 80% traning
    X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = train_test_split(
        XTemp, yTemp, test_size=0.4, random_state=0)
    X_trainTariff, X_testTariff, y_trainTariff, y_testTariff = train_test_split(
        XTariff, yTariff, test_size=0.4, random_state=0)

    # train training set
    regressorTemp = LinearRegression()
    regressorTemp.fit(X_trainTemp, y_trainTemp)

    regressorTariff = LinearRegression()
    regressorTariff.fit(X_trainTariff, y_trainTariff)

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
    global linearResultPredictedCrude

    linearResultPredictedCrude = y_predTemp.flatten()

    # show training set
    plt.scatter(X_trainTemp, y_trainTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Maintenance (Training set)')
    plt.xlabel('Quarter')
    plt.ylabel('Maintenance')
    plt.savefig('../flaskUI/static/image/lr_crudeoil_quarter_train.png')
    plt.close()

    # show test set
    plt.scatter(X_testTemp, y_testTemp, color='red')
    plt.plot(X_trainTemp, regressorTemp.predict(X_trainTemp), color='blue')
    plt.title('Quarter vs Maintenance (Test set)')
    plt.xlabel('Quarter')
    plt.ylabel('Maintenance')
    plt.savefig('../flaskUI/static/image/lr_crudeoil_quarter_test.png')
    plt.close()

    # Predict Tariff
    y_predTariff = regressorTariff.predict(X_testTariff)
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(y_testTariff, y_predTariff))
    print('Mean Squared Error:',
          metrics.mean_squared_error(y_testTariff, y_predTariff))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_testTariff, y_predTariff)))

    print("y intercept: " + str(regressorTariff.intercept_))
    print("slope: " + str(regressorTariff.coef_))

    global linearResultPredictedTarriffCrude
    a = str(regressorTariff.intercept_)
    intecept = str(a).replace('[',
                              '').replace(']',
                                          '').replace('\'',
                                                      '').replace('\"', '')
    b = str(regressorTariff.coef_)
    slope = str(b).replace('[', '').replace(']',
                                            '').replace('\'',
                                                        '').replace('\"', '')

    for x in linearResultPredictedCrude:
        predictedTariffCrude = float(intecept) + (float(slope) * x)
        predictedTariffCrude = np.round(predictedTariffCrude, 2)
        linearResultPredictedTarriffCrude.append(predictedTariffCrude)

    # linearResultPredictedTarriffCrude = y_predTariff.flatten()

    plt.scatter(X_trainTariff, y_trainTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Maintenance vs Tariff (Training set)')
    plt.xlabel('Maintenance')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_crudeoil_tariff_train.png')
    plt.close()

    plt.scatter(X_testTariff, y_testTariff, color='red')
    plt.plot(X_trainTariff,
             regressorTariff.predict(X_trainTariff),
             color='blue')
    plt.title('Maintenance vs Tariff (Test set)')
    plt.xlabel('Maintenance')
    plt.ylabel('Tariff')
    plt.savefig('../flaskUI/static/image/lr_crudeoil_tariff_test.png')
    plt.close()


if __name__ == 'main':
    # guaranteed to run on production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(hostdebug=True)
