import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0,cv.CAP_DSHOW) #카메라와 연결 시도

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frames = []

while True:
    ret,frame = cap.read()  #비디오를 구성하는 프레임 획득
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환
    canny = cv.Canny(grayframe, 0, 100)
    gray_3ch = cv.cvtColor(canny, cv.COLOR_GRAY2BGR) # 명암 영상은 1채널이라서 np.hstack을 위해 3채널로 변환
    
# 원본 이미지와 그레이스케일 이미지를 가로로 연결
    merged = np.hstack((frame, gray_3ch))
    
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    cv.imshow('Original & Grayscale', merged)
    
    key = cv.waitKey(1) # 1밀리초 동안 키보드 입력 기다림
    if key == ord('c'): # 'c' 키가 들어오면 프레임을 리스트에 추가
        frames.append(frame)
        
    elif key == ord('q'): # 'q' 키가 들어오면 루프를 빠져나감
        break



"""  
cap.release() # 카메라와 연결을 끊음
cv.destroyAllWindows()

if len(frames) > 0:

    for i in range(1,min(3,len(frames))):
        imgs = np.hstack((imgs, frames[i]))
                             
        cv.imshow('collected images', imgs)
        
        cv.waitKey()
        cv.destroyAllWindows()

print(len(frames))
print(frames[0].shape)
print(type(imgs))
print(imgs.shape)
"""