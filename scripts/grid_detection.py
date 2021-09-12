# from scripts.backtracking import sudoku_solver
import cv2
import numpy as np
import math
import sys
from scipy import ndimage
import tensorflow as tf

from numpy.core.numeric import Inf

# #read image in grayscale
# img = cv2.imread('clean_sudoku.jpg')
# if img is None:
#     sys.exit("Could not read the image.")

# model = tf.keras.models.load_model('./digit_recognizer/mnist.h5')

# Calculate how to centralize the image using its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

# Shift the image using what get_best_shift returns
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

# This function is used for seperating the digit from noise in "crop_image"
# The Sudoku board will be chopped into 9x9 small square image,
# each of those image is a "crop_image"
def largest_connected_component(image):

    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]     

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

#find four corners of sudoku contour
def find_corners(contour , num_of_corners = 4 , iterations = 100):

    # while iterations > 0:
    perimeter = cv2.arcLength(contour , True)
    epsilon = 0.1*perimeter
    poly = cv2.approxPolyDP(contour , epsilon , True)
    hull = cv2.convexHull(poly)
    if(len(hull) == num_of_corners):
        return hull
    
    return None

#find which corner is which
def find_corner_positions(corners):
                
        # A-------B
        # |       |
        # |       |
        # C-------D

    corner_pos = np.zeros((4,2) , np.float32)
    
    #finding A and D (sum of cordinated will be lowest ans highest)
    maxDist , maxPos = 0 , 0
    minDist, minPos = Inf, 0
    for i in range(4):
        if(corners[i][0] + corners[i][1] > maxDist):    #for point D
            maxDist = corners[i][0] + corners[i][1]
            maxPos = i

        if(corners[i][0] + corners[i][1] < minDist):    #for pointA
            minDist = corners[i][0] + corners[i][1]
            minPos = i


    corner_pos[3][0] = corners[maxPos][0]   #for point D
    corner_pos[3][1] = corners[maxPos][1]

    corner_pos[0][0] = corners[minPos][0]   #for point A
    corner_pos[0][1] = corners[minPos][1]

    corners = np.delete(corners , [minPos , maxPos], 0)
    
    #finding B and C
    cPos = 0
    if(corners[0][0] < corners[1][0]):
        cPos = 1

    corner_pos[2][0] = corners[cPos][0]   #for point C
    corner_pos[2][1] = corners[cPos][1]

    corner_pos[1][0] = corners[1 - cPos][0]   #for point B
    corner_pos[1][1] = corners[1 - cPos][1]

    return corner_pos

#check if this is approximately a square
def checkAngle(AB , AC , tolerance):
    cosine = np.dot(AB , AC)/(np.linalg.norm(AB) * np.linalg.norm(AC))
    angle = np.arccos(abs(cosine))*180/np.pi
    return angle > (90 - tolerance)

def check_len(AB , CD, len_tolerance_percent):
    diff_in_lengths = abs(np.linalg.norm(AB) - np.linalg.norm(CD))
    minLen = min(np.linalg.norm(AB) , np.linalg.norm(CD))
    return diff_in_lengths < len_tolerance_percent*minLen



def check_square(AB , BD , CD , AC):
    tolerance = 20      #tolerable error in angle from 90 degrees
    if not (checkAngle(AB , AC , tolerance) and checkAngle(BD , AB , tolerance) 
            and checkAngle(AC , CD , tolerance) and checkAngle(BD , CD , tolerance)):
            return False

    len_tolerance_percent = 0.2     #diff between lengths cant be grater than 0.2*len(shorter side)
    if not (check_len(AB , CD , len_tolerance_percent) and check_len(AC , BD , len_tolerance_percent)):
        return False

    return True


def extract_image(img, model):
    #https://learnopencv.com/edge-detection-using-opencv/
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img ,(5, 5) , 0)       #gaussian blur to smoothen the image to remove low intensity edges
    thres_img = cv2.adaptiveThreshold(blur_img , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV , 5 , 2)
    dialated_img = cv2.dilate(thres_img , (3, 3) , iterations= 1)

    #finding contours and the sudoku(largest contour)
    contours , hierarchy = cv2.findContours(dialated_img ,cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    bestContour = None
    for contour in contours:
        currArea = cv2.contourArea(contour)
        if(currArea > maxArea):
            maxArea = currArea
            bestContour = contour

    if bestContour is None:                 #no sudoku found
        sys.exit("No sudoku found") 


    corners = find_corners(bestContour)
    corners = np.reshape(corners , (4,2))

    corner_pos = find_corner_positions(corners)
    # corner pos -- > [A , B , C , D]
            # A-------B
            # |       |
            # |       |
            # C-------D

    # getting lengths of boundaries
    AB = corner_pos[0] - corner_pos[1]
    BD = corner_pos[1] - corner_pos[3]
    CD = corner_pos[2] - corner_pos[3]
    AC = corner_pos[0] - corner_pos[2]



    if not check_square(AB , BD , CD , AC):
        sys.exit("No sudoku found")


    width_AB = np.linalg.norm(AB)
    len_BD = np.linalg.norm(BD)
    width_CD = np.linalg.norm(CD) 
    len_AC = np.linalg.norm(AC)

    width = max(width_AB , width_CD)
    length = max(len_AC , len_BD)

    dst = np.array(
        [[0, 0],
         [0 , width],
         [length , 0],
         [length , width]], dtype= np.float32
        )


    mask = np.zeros(gray_img.shape , np.uint8)                  #create a mask for the sudoku contour
    cv2.drawContours(mask , [bestContour], 0, 255, -1)
    cv2.drawContours(mask , [bestContour], 0, 0 , 2) 

    sudoku_cropped = np.zeros(gray_img.shape)
    sudoku_cropped[mask == 255] = dialated_img[mask == 255]

    matrix = cv2.getPerspectiveTransform(corner_pos, dst)
    sudoku_img = cv2.warpPerspective(sudoku_cropped, matrix, ((int)(length) ,  (int)(width)))
    sudoku_img = cv2.erode(sudoku_img , (3,3) , iterations=2)

    #make a sudoku list
    SIZE = 9
    sudoku = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        sudoku.append(row)


    box_width = (int)(width//SIZE)
    box_length = (int)(length//SIZE)

    border_width = math.floor(box_width/10)
    border_length = math.floor(box_length/10)

    print(type(box_width))

    for i in range(9):
        for j in range(9):
            crop_image = sudoku_img[box_length*i + border_length : box_length*(i+1) - border_length , 
                                box_width*j + border_width : box_width*(j+1) - border_width]

            # There are still some boundary lines left though
            # => Remove all black lines near the edges
            # ratio = 0.6 => If 60% pixels are black, remove
            # Notice as soon as we reach a line which is not a black line, the while loop stops
            ratio = 0.6        
            # Top
            while np.sum(crop_image[0]) > (1-ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]
            # Bottom
            while np.sum(crop_image[:,-1]) > (1-ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)
            # Left
            while np.sum(crop_image[:,0]) > (1-ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
            # Right
            while np.sum(crop_image[-1]) > (1-ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]    

            #remove borders(if any)
            # crop_image = cv2.bitwise_not(crop_image)
            # Take the largestConnectedComponent (The digit), and remove all noises
            crop_image = largest_connected_component(crop_image)

            dim = 28
            # print(crop_image.shape)
            crop_image = cv2.resize(crop_image , (dim , dim))

            # If this is a white cell, set grid[i][j] to 0 and continue on the next image:

            # Criteria 1 for detecting white cell:
            # Has too little black pixels
            if crop_image.sum() >= dim**2*255 - dim * 1 * 255:
                sudoku[i][j] == 0
                continue    # Move on if we have a white cell
            # Criteria 2 for detecting white cell
            # Huge white area in the center
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]

            if center_region.sum() >= center_width * center_height * 255 - 255:
                sudoku[i][j] = 0
                continue    # Move on if we have a white cell
            
            # Now we are quite certain that this crop_image contains a number
            # Store the number of rows and cols
            rows, cols = crop_image.shape
            # Apply Binary Threshold to make digits more clear
            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY) 
            crop_image = crop_image.astype(np.uint8)
            # Centralize the image according to center of mass
            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image,shift_x,shift_y)
            crop_image = shifted
            crop_image = crop_image.reshape((1,28,28,1))

            # Recognize digits
            preds = model.predict(crop_image)
            sudoku[i][j] = np.argmax(preds,axis = 1)[0]
            # print(sudoku[i][j])

            # cv2.imshow("cropped" , crop_image)
            # cv2.imshow("ss" , crop_image)
            # cv2.imwrite('null.jpg' , crop_box)
            cv2.waitKey(0)

    return sudoku


    # cv2.imshow('img' , sudoku_cropped)
    # cv2.imshow('img_blur' , sudoku_img)
