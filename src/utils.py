import logging
import os
from datetime import datetime
import cv2
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import requests
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# model = keras.models.load_model('./models/orange_avocado_apple_model_1.h5')
model = keras.models.load_model('models/model_1.h5')


def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    # cv2.putText()
    return NotImplementedError

def predict_frame(image):
    """
    - reverse color channels
    - predict the object class

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = keras.preprocessing.image.img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    # image = preprocess_input(image)
    y_pred = model.predict(image)

    # [[0.00296858, 0.98399043, 0.00587643, 0.00716462]]
    # [0][0], [0][1], [0][2],[0][3]
    # ['apple', 'avocado', 'cactus', 'orange']
    
    if y_pred[0][0] > (y_pred[0][1] and y_pred[0][2] and y_pred[0][3]) :
        return f'I think it is an APPLE, here is my confidence level: {y_pred[0][0]}'
    elif y_pred[0][1] > (y_pred[0][0] and y_pred[0][2] and y_pred[0][3]):
        return f'I think it is an Avocado, here is my confidence level: {y_pred[0][1]}'
    elif y_pred[0][2] > (y_pred[0][0] and y_pred[0][1] and y_pred[0][3]):
        return f'I think it is a Cactus, here is my confidence level: {y_pred[0][2]}'
    else:
        return f'I think it is an Orange, here is my confidence level: {y_pred[0][3]}'

