"""
This script runs the MelaDetect application using a development server.
"""

from os import environ
from flask import Flask, render_template, request, jsonify
from MelaDetect import app
from MelaDetect import classify_image
import tensorflow as tf
import numpy as np
import cv2
import os

@app.route('/', methods=['POST', 'GET'])
def meladetect():
        filePath = os.path.dirname(os.path.abspath(__file__))
        if request.method == 'POST':
            image_to_process = []
            classify = classify_image.classify_image()
            image = request.files['file']
            #image.save does not work in Visual Studio, images are left blank?
            image.save(filePath + '\\MelaDetect' + '\\uploads' + '\\' + image.filename)
            #image.save('C:\\Users\\arron\\Machine-Learning\\MelaDetect\\uploads\\' + image.filename)
            img_name = image.filename.split('.')[0]
            image = classify.process_images(image)


            #image_to_process.append([np.array(image), img_name])
            #x_batch = classify.process_images(image)
            x_batch = np.array([i[0] for i in image]).reshape(-1, 122, 122, 3)
            y_pred = app.graph.get_tensor_by_name("FullyConnected_1/Softmax:0")
            ## Let's feed the images to the input placeholders
            listOfNodes = [n.name for n in app.graph.as_graph_def().node]
            #print('{}').format(listOfNodes)
            x = app.graph.get_tensor_by_name("input/X:0")
            y_test_images = np.zeros((1, 2))
            sess= tf.Session(graph=app.graph)
            ### Creating the feed_dict that is required to be fed to calculate y_pred 
            feed_dict_testing = {x: x_batch}
            result=sess.run(y_pred, feed_dict=feed_dict_testing)
            result = result.tolist()
            print(result)
            return render_template('prediction.html', prediction = result) 
            #return jsonify(result)
            
        return render_template('index.html')   

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
