# 컴퓨터 비전

컴퓨터 비전 4주차 실습 과제 1번


![image](https://github.com/user-attachments/assets/61f05df6-3f5d-409a-af70-19432e8c17df)


cv.imread()를 사용하여 이미지 불러오고, Cv.cvtColor()를 사용하여 그레이스케일로 변환

    img = cv.imread('soccer.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.Sobel()을 사용하여X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출

    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize = 3) # 소벨 연산자 적용
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize = 3)

cv.magnitude()를 사용하여 에지강도를계산

    edge_strength = cv.magnitude(grad_x, grad_y)
    edge_strength = cv.convertScaleAbs(edge_strength)  # uint8 변환


matplotlib를 사용하여 원본이미지와 에지강도이미지를 나란히 시각화
```python
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
```

전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize = 3) 
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize = 3)

edge_strength = cv.magnitude(grad_x, grad_y)
edge_strength = cv.convertScaleAbs(edge_strength)

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
```

cv.convertScaleAbs가 없으면 색이 좀 연하게 나오는데 그 이유가
이미지의 각 픽셀 값이 부호 없는 8비트 정수 (0에서 255 사이)여야 하기 때문이다. 
OpenCV는 이미지를 표시하거나 저장할 때 부호 없는 8비트 값을 사용한다. 
만약 부호가 있는 값으로 그대로 두면 이미지로 나타낼 때 오류가 발생하거나 예상치 못한 결과가 나올 수 있다.


결과 화면
cv.convertScaleAbs 있는 경우

![image](https://github.com/user-attachments/assets/e18cdb8b-78be-4698-b05e-522f08e84ab7)

cv.convertScaleAbs 없는 경우

![image](https://github.com/user-attachments/assets/b35dbd7d-b6ee-4f44-91f2-ed8163969981)



---

컴퓨터 비전 4주차 실습 과제 2번

![image](https://github.com/user-attachments/assets/51240ff7-8d3b-450d-949b-8f37a4287dd1)


cv.Canny()를 사용하여에지맵을생성

    edges = cv.Canny(gray, 100, 200)  # Tlow=100, Thigh=200 설정

cv.HoughLinesP()를 사용하여 직선을검출

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=5)

cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림

    results = img_rgb.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(results, (x1, y1), (x2, y2), (255, 0, 0), 2) 

matplotlib를 사용하여 원본이미지와 직선이 그려진 이미지를 나란히 시각화
```python
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(results)
plt.title('Detected Lines')
plt.axis('off')

plt.show()
```

전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)  # Tlow=100, Thigh=200 설정

lines = cv.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=5)

results = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(results, (x1, y1), (x2, y2), (255, 0, 0), 2) 



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(results)
plt.title('Detected Lines')
plt.axis('off')

plt.show()
```

결과 화면

![image](https://github.com/user-attachments/assets/5569d426-f47d-4d2d-b0c0-4900200a8c5b)


---

컴퓨터 비전 4주차 실습 과제 3번

![image](https://github.com/user-attachments/assets/6a8734fe-6c8b-4700-be50-9e7e530fdda7)


초기사각형영역은(x, y, width, height) 형식으로 설정

    rc = (50, 50, src.shape[1] - 100, src.shape[0] - 100)  # 이 값은 필요에 맞게 조정 가능


마스크를 사용하여 원본이미지에서 배경을 제거

    mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    dst = src * mask2[:, :, np.newaxis]

mask 값이 cv2.GC_BGD (배경) 또는 cv2.GC_PR_BGD (배경 확신)인 픽셀을 0으로 설정하고, 나머지 전경 픽셀을 유지하여 배경을 제거한 이미지를 dst에 저장합니다.


matplotlib를 사용하여원본이미지, 마스크이미지, 배경제거이미지세개를나란히시각화
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(src)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask2, cmap='gray')
axes[1].set_title('Mask Image')
axes[1].axis('off')

axes[2].imshow(dst)
axes[2].set_title('Foreground Removed')
axes[2].axis('off')

plt.show()
```

cv.grabCut()를 사용하여 대화식분할을 수행
```python
cv2.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)  
        cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1) 
        
    elif event == cv2.EVENT_RBUTTONDOWN:  
        cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)  
        cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)  
        cv2.imshow('dst', dst)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_LBUTTONDOWN:
            cv2.circle(dst, (x,y), 3, (255, 0, 0), -1)
            cv2.circle(mask, (x,y), 3, cv2.GC_FGD, -1)
            cv2.imshow('dst', dst)
        elif flags & cv2.EVENT_RBUTTONDOWN:
            cv2.circle(dst, (x,y), 3, (0, 0, 255), -1)
            cv2.circle(mask, (x,y), 3, cv2.GC_BGD, -1)
            cv2.imshow('dst', dst)
            
cv2.setMouseCallback('dst', on_mouse)
```

전체 코드
```python
import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt

src = skimage.data.coffee()

rc = (50, 50, src.shape[1] - 100, src.shape[0] - 100)  # 이 값은 필요에 맞게 조정 가능

mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iterCount = 1
mode = cv2.GC_INIT_WITH_RECT

cv2.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]
cv2.imshow('dst', dst) 

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)  
        cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1) 
        
    elif event == cv2.EVENT_RBUTTONDOWN:  
        cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)  
        cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)  
        cv2.imshow('dst', dst)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_LBUTTONDOWN:
            cv2.circle(dst, (x,y), 3, (255, 0, 0), -1)
            cv2.circle(mask, (x,y), 3, cv2.GC_FGD, -1)
            cv2.imshow('dst', dst)
        elif flags & cv2.EVENT_RBUTTONDOWN:
            cv2.circle(dst, (x,y), 3, (0, 0, 255), -1)
            cv2.circle(mask, (x,y), 3, cv2.GC_BGD, -1)
            cv2.imshow('dst', dst)
            
cv2.setMouseCallback('dst', on_mouse)


while True:
    key = cv2.waitKey()
    if key == 13:  
        cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        dst = src * mask2[:, :, np.newaxis]
        cv2.imshow('dst', dst)
    elif key == 27:  
        break

mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(src)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask2, cmap='gray')
axes[1].set_title('Mask Image')
axes[1].axis('off')

axes[2].imshow(dst)
axes[2].set_title('Foreground Removed')
axes[2].axis('off')

plt.show()
```

결과 화면

초기 화면
![image](https://github.com/user-attachments/assets/fbfd0794-6533-4068-b0de-44e74c444372)

보이게할 부분 드래그
![image](https://github.com/user-attachments/assets/1743022f-9810-4d73-9131-479f942aaaac)

Enter 키 누른후 화면
![image](https://github.com/user-attachments/assets/e0f2bca0-6111-4827-b560-3e5642a46f77)

ESC 누른후 원본, 마스크, 배경제거 이미지 생성
![image](https://github.com/user-attachments/assets/d810b351-7ffa-442c-80de-fc8bc8195303)
