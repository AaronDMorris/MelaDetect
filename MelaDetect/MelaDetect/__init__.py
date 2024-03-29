"""
The flask application package.
"""

from flask import Flask
from MelaDetect import classify_image
import os

filePath = os.path.dirname(os.path.abspath(__file__))
classify = classify_image.classify_image()
app = Flask(__name__)
app.graph = classify.load_graph(filePath + '\\12-03-frozen_model.pb')

import MelaDetect.views
import tensorflow as tf
