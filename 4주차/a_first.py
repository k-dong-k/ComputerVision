import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize = 3) 
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize = 3)

edge_strength = cv.magnitude(grad_x, grad_y)
#edge_strength = cv.convertScaleAbs(edge_strength) 

'''
cv.imshow('Original', gray)
cv.imshow('sobelx', grad_x)
cv.imshow('sobely', grad_y)
cv.imshow('edge strength', edge_strength)
'''
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
plt.axis('off')

plt.show()