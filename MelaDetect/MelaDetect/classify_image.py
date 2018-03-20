class classify_image(object):
    """description of class"""
    

    def __init__(self):
        pass

    def load_graph(self, trained_model):
        import tensorflow as tf
        with tf.gfile.GFile(trained_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
        
        return graph

    def process_images(self, image):
        import cv2
        import numpy as np
        image_array = []
        #image = cv2.resize(cv2.imread('C:\\Users\\arron\\Machine-Learning\\MelaDetect\\uploads\\' + image.filename, cv2.IMREAD_COLOR), (122, 122))
        #image = cv2.fastNlMeansDenoising(img, 5, 7)
        #image = cv2.resize(cv2.imread('E:\\MelaDetect\\alldata_testdata_malignant\\' + '116.jpg', cv2.IMREAD_COLOR), (122, 122))

        image = cv2.resize(cv2.imread('E:\\MelaDetect\\alldata_testdata_benign\\' + '1489.jpg', cv2.IMREAD_COLOR), (122, 122))
        image_array.append([np.array(image)])
        
        return image_array      


