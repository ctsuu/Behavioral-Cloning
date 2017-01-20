import argparse
import base64
import json
from io import BytesIO

import eventlet.wsgi
import numpy as np
from numpy import arange
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
from keras.models import model_from_json

import helper

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#apply 10 frame moving average
i=0
j=0
smooth = []
moving_range = 3
moving_avg = arange(-1,1.1,2/moving_range) 
print ('Apply '+str(len(moving_avg))+' frame moving average to steering records')


"""
def crop(image, top_cropping_percent):
    assert 0 <= top_cropping_percent < 1.0, 'top_cropping_percent should be between zero and one'
    percent = int(np.ceil(image.shape[0] * top_cropping_percent))
    return image[percent:, :, :]
"""

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]

    # The current throttle of the car
    throttle = data["throttle"]

    # The current speed of the car
    speed = data["speed"]

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    image_array = helper.crop(image_array, 0.28, 0.17)
    image_array = helper.resize(image_array, new_dim=(64, 64))

    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    #Apply number of frames moving average to steering records 
    global i
    moving_avg[i%moving_range] = steering_angle
    smooth_angle=np.mean(moving_avg)*1.05
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.23


    print('{:.5f}, {:.2f}'.format(smooth_angle, throttle))
    i = i+1

    send_control(smooth_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
