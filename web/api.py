# system
import io

# import
import numpy as np
import flask
from flask import jsonify
from keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

# self
from models.steganography.d_2 import Steganography2D

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)

model = None


def load_model():
    """
    load the political sentiment deep learning model
    being used and set it as a global.

    :return: the model
    """
    global model

    # note this model at ./bin/main might not be here
    # if it is not and you are trying to do it, let me know.

    model = Steganography2D(dir='./bin/main')
    model.load()
    print('model loaded')


load_model()


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/encode', methods=['POST'])
def encode():
    pass
    # TODO


@app.route('/decode', methods=['POST'])
def decode():
    pass
    # TODO

if __name__ == "__main__":
    # load_model()
    app.run()
