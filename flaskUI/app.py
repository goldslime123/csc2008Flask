from flask import Flask, render_template
import os

app = Flask(__name__,template_folder="templates")
IS_DEV = app.env == 'development'


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == 'main':
     # guaranteed to not be run on a production server
    assert os.path.exists('.env')  # for other environment variables...
    os.environ['FLASK_ENV'] = 'development'  # HARD CODE since default is production
    app.run(debug=True)