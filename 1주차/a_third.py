import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')
    
# Step 1. 마우스 클릭한 곳에 직사각형 그리기
def draw(event, x, y, flags, papram):
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭 했을 떄
        cv.rectangle(img, (x,y), (x+200, y+200), (0, 0, 255), 2)
    elif event == cv.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼 클릭 했을 떄
        cv.rectangle(img, (x,y), (x+100, y+100), (255, 0, 0), 2)
    cv.imshow('Drawing',img)
    
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw) # Drawing 윈도우에 draw 콜백 함수 지정

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

""" 
cv.rectangle(img, (100,200), (300,450), (0,0,255), 2) #직사각형 그리기
cv.putText(img, 'mouse', (200,300), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #글씨 쓰기

cv.imshow('Draw',img)

cv.waitKey(0)
cv.destroyAllWindows()
"""