# 컴퓨터 비전

컴퓨터 비전 1주차 실습 과제 1번

![image](https://github.com/user-attachments/assets/e5581abb-b905-458d-8495-ae00082d38c8)

cv.cvtColor() 함수를 사용해 이미지를 그레이 스케일로 변환 후 저장

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('soccer_gray.jpg', gray) 


원본 이미지와 그레이 스케일 이미지를 가로로 출력하기위해 채널수 맞춘 후 연결

    gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    merged = np.hstack((img, gray_3ch))


전체 코드
```python
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
```

결과 화면

![image](https://github.com/user-attachments/assets/d54161ed-f290-448f-be32-7802b5c804e3)

---

컴퓨터 비전 1주차 실습 과제 2번

![image](https://github.com/user-attachments/assets/f33cb9ff-c729-4263-9b7a-88750fa5f50e)


cv.VideoCapture()를 사용해 웹켐 영상 로드

    cap = cv.VideoCapture(0,cv.CAP_DSHOW) #카메라와 연결 시도


각 프레임을 그레이스케일로 변환, cv.Canny() 함수를 사용해 에지 검출 후 가로로 연결하여 화면에 출력
```python
ret,frame = cap.read()  #비디오를 구성하는 프레임 획득
grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환
canny = cv.Canny(grayframe, 0, 100)
gray_3ch = cv.cvtColor(canny, cv.COLOR_GRAY2BGR) # 명암 영상은 1채널이라서 np.hstack을 위해 3채널로 변환  
merged = np.hstack((frame, gray_3ch)) # 원본 이미지와 그레이스케일 이미지를 가로로 연결
```

q 누르면 영상 종료
```python
if key == ord('q'): # 'q' 키가 들어오면 루프를 빠져나감
    break
```

전체 코드
'''python
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
 
    if key == ord('q'): # 'q' 키가 들어오면 루프를 빠져나감
        break
```

결과 화면

![image](https://github.com/user-attachments/assets/aa04851c-8912-4fa2-b64c-66184a091217)


---

컴퓨터 비전 1주차 실습 과제 3번

![image](https://github.com/user-attachments/assets/47121a92-f180-495f-9382-be735e7db5de)

cv.SetMouseCallback()을 사용 마우스 이벤트 처리

    cv.setMouseCallback('Drawing', draw)


사용자가 클릭한 시작점에서 드래그 하여 사각형 그리며 영역 선택 마우스 놓으면 해당영역 잘라내어 별도의 창 띄움
```python
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
```


r 키를 누르면 영역 선택 리셋, s 키 누르면 선택한 영역 이미지 저장
```python
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
```

결과 화면

![image](https://github.com/user-attachments/assets/48cdb493-445c-48fa-82a7-3d268eab0b44)

