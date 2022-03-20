from flask import Flask, render_template
import os

app = Flask(__name__, template_folder="Website")
IS_DEV = app.env == 'development'


@app.route('/')
def index():
    #This is where we import our data from database
    data = [
        ("2015-Q1", 23.29),
        ("2015-Q2", 20.87),
        ("2015-Q3", 22.41),
        ("2015-Q4", 20.35),
        ("2016-Q1", 19.5),
        ("2016-Q2", 17.68),
        ("2016-Q3", 19.28),
        ("2016-Q4", 19.13),
        ("2017-Q1", 20.2),
    ]
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    return render_template("index.html", labels=labels, values=values)


if __name__ == 'main':
    # guaranteed to not be run on a production server
    assert os.path.exists('.env')  # for other environment variables...
    # HARD CODE since default is production
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
