import cv2
import sys

#read image in grayscale
img = cv2.imread('final.jpg' , cv2.IMREAD_GRAYSCALE)

# if img is None:
#     sys.exit("Could not read the image.")

cv2.imshow('img' , img)
cv2.waitKey(0)