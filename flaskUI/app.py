from flask import Flask, render_template
import os

# MariaDB Imports
# import mariadb
# import sys

app = Flask(__name__, template_folder="Website")
IS_DEV = app.env == 'development'


@app.route('/')
def index():

    # list = []

    # # Connect to MariaDB Platform
    # try:
    #     conn = mariadb.connect(
    #         user="root",
    #         password="database",
    #         host="localhost",
    #         port=3306,
    #         database="csvfiles"

    #     )
    #     print("Successfully connected", file=sys.stderr)
    # except mariadb.Error as e:
    #     print(f"Error connecting to MariaDB Platform: {e}", file=sys.stderr)
    #     sys.exit(1)

    # # Get Cursor
    # cur = conn.cursor()
    # cur.execute(
    # "SELECT t.quarter, t.temperature, t.tariff_per_kwh, c.price_per_barrel FROM temp t, crudeoil c WHERE t.quarter=c.quarter;")

    # for i in cur:
    #     list.append(i)

    # data = [(item[0], float(item[1]), float(item[2]), float(item[3])) for item in list]
    # for i in data:
    #     print(i, file=sys.stderr)

    # #from database
    # labels = [row[0] for row in data]
    # electricPrice = [row[2] for row in data]
    # crudePrice = [row[3] for row in data]
    # return render_template("index.html", labels=labels, electricPrice=electricPrice, crudePrice=crudePrice )


    # This is where we import our data from database
    data = [
        ("2015-Q1", 23.29, 25.6, 27.4, 21.9),
        ("2015-Q2", 20.87, 25.6, 28.8, 21.9),
        ("2015-Q3", 22.41, 25.6, 28.7, 21.9),
        ("2015-Q4", 20.35, 27.9, 28.1, 21.9),
        ("2016-Q1", 19.5, 27.9, 28.4, 24.4),
        ("2016-Q2", 17.68, 27.9, 29.1, 24.4),
        ("2016-Q3", 19.28, 27.9, 27.7, 24.4),
        ("2016-Q4", 19.13, 27.9, 27.1, 24.4),
        ("2017-Q1", 20.2, 24.2, 28.3, 24.9),
    ]
    
    labels = [row[0] for row in data]
    electricPrice = [row[1] for row in data]
    crudePrice = [row[2] for row in data]
    temperature = [row[3] for row in data]
    maintenance = [row[4] for row in data]
    return render_template("index.html", labels=labels, electricPrice=electricPrice, crudePrice=crudePrice, temperature=temperature, maintenance=maintenance )


if __name__ == 'main':
    # guaranteed to not be run on a production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
