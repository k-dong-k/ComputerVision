import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('img1.jpg')[190:350, 440:560] #버스를 크롭하여 모델 영상으로 사용
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp1,des1 = sift.detectAndCompute(gray1, None)
kp2,des2 = sift.detectAndCompute(gray2, None)

bf_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
knn_match = bf_matcher.knnMatch(des1, des2, 2)  # 최근접 2개 

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match :
    if(nearest1.distance/nearest2.distance) < T :
        good_match.append(nearest1)
        
points1 = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

H, _ = cv.findHomography(points1, points2, cv.RANSAC)

h2, w2 = img2.shape[:2]
img1_warped = cv.warpPerspective(img1, H, (w2, h2))

plt.figure(figsize=(12, 6))
    
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.title("Original Cropped Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img1_warped, cv.COLOR_BGR2RGB))
plt.title("Homography Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.title("Image")
plt.axis('off')

plt.show()

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("Result")
plt.axis('off')
plt.show()