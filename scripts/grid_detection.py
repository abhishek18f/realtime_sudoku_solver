# from scripts.backtracking import sudoku_solver
import cv2
import numpy as np
import math
import sys

from numpy.core.numeric import Inf

#read image in grayscale
img = cv2.imread('clean_sudoku.jpg')
if img is None:
    sys.exit("Could not read the image.")

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

corners = find_corners(bestContour)
corners = np.reshape(corners , (4,2))

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

for i in range(1):
    for j in range(1,2):
        crop_box = sudoku_img[box_length*i + border_length : box_length*(i+1) - border_length , 
                            box_width*j + border_width : box_width*(j+1) - border_width]

        #remove borders(if any)
        

        # crop_box = cv2.bitwise_not(crop_box)
        cv2.imshow('img' , sudoku_img)
        cv2.imshow("cropped" , crop_box)
        cv2.waitKey(0)


# cv2.imshow('img' , sudoku_cropped)
# cv2.imshow('img_blur' , sudoku_img)
