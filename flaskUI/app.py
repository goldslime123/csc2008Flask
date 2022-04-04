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

# Predict tariff based on tempeature
predictedTemp = 0
predictedTarriff_temp = 0

#################################################################################
############################ HomePage ###########################################
#################################################################################
@app.route('/')
def home():
    
    #load gif image:
    gifImage = os.path.join(app.config['UPLOAD_FOLDER'],
                                         'LightBulb.gif')
    
    return render_template("home.html", gifImage=gifImage)

#################################################################################
############################ Temperature Page ###################################
#################################################################################
@app.route('/temperature')
def temperature():

    list = []
    conn = None

    # linear regression training
    LR_temperature()
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
            password="3b79c1f6062e3164cb523ea49ade123ccc4d25a86f7fa9c7e2b42921d0f55831")

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

        linearData = [(item[0], float(item[1]), float(item[2]))
                       for item in list]

        nonLinearData = [(item[0], float(item[1]), float(item[2]))
                          for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear
        l_labels = [row[0] for row in linearData]
        l_labels.append('2022.1')
        l_temperature = [row[1] for row in linearData]
        l_electricPrice = [row[2] for row in linearData]

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
            lr_temp_quarter_train=lr_temp_quarter_train,
            lr_temp_quarter_test=lr_temp_quarter_test,
            lr_temp_tariff_train=lr_temp_tariff_train,
            lr_temp_tariff_test=lr_temp_tariff_test,
        )

#################################################################################
############################ Crude Oil Page #####################################
#################################################################################
@app.route('/crudeoil')
def crudeoil():

    list = []
    conn = None

    # Connect to postgresql Platform
    try:
        conn = psycopg2.connect(
            host="ec2-54-173-77-184.compute-1.amazonaws.com",
            database="d2v75ijfptfl5f",
            user="jkbetvbzvsivpk",
            password="3b79c1f6062e3164cb523ea49ade123ccc4d25a86f7fa9c7e2b42921d0f55831")

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

        linearData = [(item[0], float(item[2]), float(item[3]))
                       for item in list]

        nonLinearData = [(item[0], float(item[2]), float(item[3]))
                          for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear
        l_labels = [row[0] for row in linearData]
        l_labels.append('2022.1')
        l_electricPrice = [row[1] for row in linearData]
        l_crudePrice = [row[2] for row in linearData]

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
        )

#################################################################################
############################ Maintenance Page ###################################
#################################################################################
@app.route('/maintenance')
def maintenance():
    
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

        # show linear and non linear
        cur = conn.cursor()
        cur.execute(
            "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;"
        )

        for i in cur:
            list.append(i)

        linearData = [(item[0], float(item[2]), float(item[4])) for item in list]
        
        nonLinearData = [(item[0], float(item[2]), float(item[4])) for item in list]
        # for i in data:
        #     print(i, file=sys.stderr)

        # Linear
        l_labels = [row[0] for row in linearData]
        l_labels.append('2022.1')
        l_electricPrice = [row[1] for row in linearData]
        l_maintenance = [row[2] for row in linearData]
        
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
        )

#################################################################################
############################ Linear ML Functions ################################
#################################################################################
def LR_temperature():
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
        
    df = pd.DataFrame(data) 
            
    # saving the dataframe 
    df.to_csv('temp/temperature.csv',header=['quarter', 'temperature', 'tariff_per_kwh'], index=False)

    # import data set
    dataset= pd.read_csv('temp/temperature.csv')

    XTemp = dataset['quarter'].values.reshape(-1, 1)
    yTemp = dataset['temperature'].values.reshape(-1, 1)

    XTariff = dataset['temperature'].values.reshape(-1, 1)
    yTariff = dataset['tariff_per_kwh'].values.reshape(-1, 1)

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
    global predictedTemp
    predictedTemp = regressorTemp.intercept_ + (regressorTemp.coef_ * 2022.1)
    print("Predicted Temperature: 2022 Quarter 1: " + str(predictedTemp))

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

    global predictedTarriff_temp
    predictedTarriff_temp = regressorTariff.intercept_ + (
        regressorTariff.coef_ * predictedTemp)
    print("Predicted Tariff: 2022 Quarter 1: " + str(predictedTarriff_temp))

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


if __name__ == 'main':
    # guaranteed to run on production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
