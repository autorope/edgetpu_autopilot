"""A demo to classify image."""
from PIL import Image
import os
import time
import tempfile

import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense


from engine import AutopilotEngine


def run_model(model_path, image_path):
    start_time = time.time()

    # Initialize engine.
    engine = AutopilotEngine(model_path)
    print(f'laoded model {time.time() - start_time}')

    # Run inference.
    img = Image.open(image_path)
    print(f'opened image {time.time() - start_time}')

    for _ in range(10):
        result = engine.ClassifyWithImage(img)
        print(f'classify image {time.time() - start_time}')
        print(result)

    return result


def convert_keras_to_tflite(keras_model, tflite_model_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        keras_model_path = os.path.join(tmp_dir, 'keras_model.h5')
        model.save(keras_model_path)
        print('creating converter')
        converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model_path)

        print('converting model')
        tflite_model = converter.convert()
        print('writing file')


        with open(tflite_model_path, "wb") as fp:
            fp.write(tflite_model)
            fp.flush()
            return tflite_model_path



def get_keras_model():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model


if __name__ == '__main__':

    package_path = os.path.dirname(__file__)
    image_path = '/home/wroscoe/data/diyrobocar_races/2018-03_aws_reinvent/tub_1_17-11-28/1_cam-image_array_.jpg'

    model = get_keras_model()

    tflite_model_name = 'keras_to_tflite_model'
    tflite_model_path = os.path.join(package_path, tflite_model_name + '.tflite')
    convert_keras_to_tflite(model, tflite_model_path)

    #result = run_model(compiled_model_path, image_path)
    #print(result)