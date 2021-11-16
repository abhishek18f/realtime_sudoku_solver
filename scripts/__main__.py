from backtracking import sudoku_solver
import tensorflow as tf
from grid_detection import *
import cv2

img = cv2.imread('clean_sudoku.jpg')
model = tf.keras.models.load_model('./digit_recognizer/mnist.h5')

def solver(img):
    sudoku = extract_image(img , model)
    solved_sudoku = sudoku_solver(sudoku)
    print(solved_sudoku)
    # model.summary()

solver(img)

    

