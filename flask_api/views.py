# import bash.bosh

from flask import Blueprint, jsonify, request
import tensorflow as tf
from .scripts.backtracking import sudoku_solver
from .scripts.grid_detection import *
from .scripts.solver import solver


model = tf.keras.models.load_model('./scripts/digit_recognizer/mnist.h5')

main =Blueprint('main', __name__)

@main.route('/process', methods = ['POST'])
def process():
    files = request.files
    file = files.get('file') 

    print(type(file))
    return jsonify({
        'success': True,
        'file': files
    })
