import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

def draw(event, x, y, flags, papram):
    global ix,iy
    global roi
    
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭했을때 초기 위치 저장
        ix, iy = x,y
    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 땟을떄 직사각형 그리기
        cv.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)
        roi = img[iy:y, ix:x]
        cv.imshow("roi",roi)
        
    cv.imshow('Drawing', img)
    
    
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    keyboard = cv.waitKey(1)
    if keyboard ==ord('q'):
        cv.destroyAllWindows()
        break
    elif keyboard == ord('r'):
        img = cv.imread('soccer.jpg')
        cv.imshow('Drawing', img)
    elif keyboard == ord('s'):
        cv.imwrite('roi.jpg', roi)
