import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

gray = cv.drawKeypoints(gray, kp, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('SIFT Image')
plt.axis('off')

plt.show()