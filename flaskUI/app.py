from flask import Flask, render_template
import os

# postgres
import psycopg2
import sys

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
            password="3b79c1f6062e3164cb523ea49ade123ccc4d25a86f7fa9c7e2b42921d0f55831") 
        
        print("Successfully connected", file=sys.stderr)

    except Exception as error:
        print("Error connecting to Postgres Platform: {}".format(error))

    # Get Cursor
    if conn is not None:
    
        cur = conn.cursor()
        cur.execute(
        "SELECT w.quarter, w.temperature, t.tariff_per_kwh, c.price_per_barrel, m.cost FROM weather w, tariff t, crudeoil c, maintenance m WHERE w.quarter=c.quarter and  w.quarter=t.quarter and m.quarter=w.quarter;")


        for i in cur:
            list.append(i)

        data = [(item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4])) for item in list]
        for i in data:
            print(i, file=sys.stderr)

        #from database
        labels = [row[0] for row in data]
        temperature = [row[1] for row in data]
        electricPrice = [row[2] for row in data]
        crudePrice = [row[3] for row in data]
        maintenance = [row[4] for row in data]
        return render_template("index.html", labels=labels, electricPrice=electricPrice, crudePrice=crudePrice,temperature=temperature, maintenance=maintenance )


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


if __name__ == 'main':
    # guaranteed to not be run on a production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
