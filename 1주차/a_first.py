import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환
cv.imwrite('soccer_gray.jpg', gray)  # 영상을 파일에 저장

# 명암 영상은 1채널이라서 np.hstack을 위해 3채널로 변환
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 원본 이미지와 그레이스케일 이미지를 가로로 연결
merged = np.hstack((img, gray_3ch))

cv.imshow('Original & Grayscale', merged)

cv.waitKey(0)
cv.destroyAllWindows()

print(type(img))
print(img.shape)

# cv.imshow('Image Display', img)

"""
Step 1 (0,0)의 BGR값 출력력

print(img[0,0,0], img[0,0,1], img[0,0,2])
"""

"""
Step 2 영상 형태 변환하고 크기 축소

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #BGR 컬러 영상을 명암 영상으로 변환
gray_small=cv.resize(gray, dsize=(0,0) ,fx=0.5, fy=0.5) #반으로 축소

cv.imwrite('soccer_gray.jpg',gray) #영상을 파일에 저장
cv.imwrite('soccer_gray_small.jpg', gray_small)

cv.imshow('Color image', img)
cv.imshow('Gray image', gray)
cv.imshow('Gray image small',gray_small)
"""
