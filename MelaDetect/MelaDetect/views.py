"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from MelaDetect import app

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/prediction')
def prediction():
    """Renders the prediction page."""
    return render_template(
        'prediction.html',
        title='Your Prediction',
        year=datetime.now().year,
        message='Your prediction page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
