# 1. 소벨 에지 검출 및 결과 시각화

📢 설명

* 이미지를그레이스케일로변환합니다.

* 소벨(Sobel) 필터를 사용하여 X축과 Y축 방향의 에지를 검출합니다.

* 검출된에지강도(edge strength) 이미지를 시각화합니다.

📖 요구사항

* cv.imread()를 사용하여 이미지를 불러옵니다.

* cv.cvtColor()를 사용하여 그레이스케일로 변환합니다.

* cv.Sobel()을 사용하여X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출합니다.

* cv.magnitude()를 사용하여 에지강도를계산합니다.

* matplotlib를 사용하여원본이미지와에지강도이미지를나란히시각화합니다.


### cv.imread()를 사용하여 이미지 불러오고, Cv.cvtColor()를 사용하여 그레이스케일로 변환

    img = cv.imread('soccer.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


* cv.imread() 함수를 사용하여 이미지 읽음
  
* cv.COLOR_BGR2GRAY 를 사용하여 BGR -> 그레이 스케일로 변환

  
### cv.Sobel()을 사용하여X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출

    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize = 3) # 소벨 연산자 적용
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize = 3)


cv.Soobel() = 미분을 이용하여 이미지 경계를 검출

* gray -> 입력 이미지
  
* cv.CV_64F -> 출력 이미지의 데이터 타입 (64비트 실수형, 음수값 포함 가능)
  
* 1, 0 -> X축 방향 1차 미분 수평 방향의 경계를 검출

* 0, 1 -> Y푹 방향 1차 미분 수직 방향의 경계를 검출

* ksize = 3 -> 커널 크기


### cv.magnitude()를 사용하여 에지강도를계산

    edge_strength = cv.magnitude(grad_x, grad_y)
    edge_strength = cv.convertScaleAbs(edge_strength)  # uint8 변환

    
* cv.magnitude()는 벡터의 크기(에지 강도)를 계산하는 함수

* 수평 방향, 수직 방향의 기울기 -> 피타고라스 정리를 이용하여 에지의 강도 계산

![image](https://github.com/user-attachments/assets/f3cdd778-c12e-47e4-a652-8f9b95c35194)


* cv.convertScaleAbs()는 음수 값을 제거하고(절댓값 변환), uint8 형식으로 변환


### matplotlib를 사용하여 원본이미지와 에지강도이미지를 나란히 시각화
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

### 전체 코드
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


### 결과 화면
cv.convertScaleAbs 있는 경우

![image](https://github.com/user-attachments/assets/e18cdb8b-78be-4698-b05e-522f08e84ab7)

cv.convertScaleAbs 없는 경우

![image](https://github.com/user-attachments/assets/b35dbd7d-b6ee-4f44-91f2-ed8163969981)


---


# 2. 캐니 에지 및 허프 변환을 이용한 직선 검출

📢 설명

* 캐니(Canny) 에지검출을 사용하여 에지맵을 생성합니다.

* 허프변환(Hough Transform)을 사용하여 이미지에서 직선을 검출합니다

* 검출된 직선을 원본이미지에 빨간색으로 표시합니다.

📖 요구사항

* cv.Canny()를 사용하여 에지맵을 생성합니다.

* cv.HoughLinesP()를 사용하여 직선을 검출합니다.

* cv.line()을 사용하여 검출된 직선을 원본이미지에 그립니다.

* matplotlib를 사용하여 원본이미지와 직선이 그려진 이미지를 나란히 시각화합니다.

  

### cv.Canny()를 사용하여에지맵을생성

    edges = cv.Canny(gray, 100, 200)  # Tlow=100, Thigh=200 설정

* gray -> 입력 이미지
  
* 100 -> 하한 임계값 (이 값보다 작은 그래디언트는 에지로 간주 X)

* 200 -> 상한 임계값 (이 값보다 큰 그래디언트는 확실한 에지로 간주)


### cv.HoughLinesP()를 사용하여 직선을검출

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=5)

* cv.Canny 를 통해 edges(에지 맵)을 입력으로 받음

* 1 -> 해상도 픽셀단위

* np.pi / 180 -> 직선을 찾을 때 각도를 몇 도 단위로 나눌지 결정

* 160 -> threshold 직선으로 인정받기 위해 누적되어야하는 최소한의 투표 수 160표를 받은 직선만 최종 검출


### cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림

    results = img_rgb.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(results, (x1, y1), (x2, y2), (255, 0, 0), 2) 

* 검출한 직선이 있을때 실행

* 직선정보가 리스트로 저장

* cv.line( 직선을 그릴 대상 이미지, 시작점 좌표, 끝점 좌표, 파란색, 선의두께)를 나타냄


### matplotlib를 사용하여 원본이미지와 직선이 그려진 이미지를 나란히 시각화
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

### 전체 코드
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

### 결과 화면

![image](https://github.com/user-attachments/assets/5569d426-f47d-4d2d-b0c0-4900200a8c5b)


---

# 3. GrabCut을 이용한대화식영역분할및객체추출

📢 설명

* 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체를 추출합니다.

* 객체 추출 결과를 마스크 형태로 시각화 합니다.

* 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력합니다.


📖 요구사항

* cv.grabCut()를 사용하여 대화식분할을 수행합니다.

* 초기 사각형 영역은(x, y, width, height) 형식으로 설정하세요.

* 마스크를 사용하여 원본이미지에서 배경을 제거합니다.

* matplotlib를 사용하여 원본이미지, 마스크이미지, 배경제거이미지 세개를 나란히 시각화합니다.


### 초기사각형영역은(x, y, width, height) 형식으로 설정

    rc = (50, 50, src.shape[1] - 100, src.shape[0] - 100)  # 이 값은 필요에 맞게 조정 가능


* src.shape[1] - 100 -> 원본이미지의 가로크기에서 100 픽셀을 뺀 값

* src.shape[0] - 100 -> 원본이미지의 세로 크기에서 100 픽셀을 뺸 값


### 마스크를 사용하여 원본이미지에서 배경을 제거

    mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    dst = src * mask2[:, :, np.newaxis]


mask 값이 cv2.GC_BGD (배경) 또는 cv2.GC_PR_BGD (배경 확신)인 픽셀을 0으로 설정하고, 나머지 전경 픽셀을 유지하여 배경을 제거한 이미지를 dst에 저장


### matplotlib를 사용하여원본이미지, 마스크이미지, 배경제거이미지세개를나란히시각화
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

### cv.grabCut()를 사용하여 대화식분할을 수행
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

### 1. 마우스로 전경과 배경을 지정
- **왼쪽 클릭** (`cv2.EVENT_LBUTTONDOWN`)을 통해 **전경(Foreground)**을 선택합니다.
- **오른쪽 클릭** (`cv2.EVENT_RBUTTONDOWN`)을 통해 **배경(Background)**을 선택합니다.

이 정보는 **mask 배열**에 기록되며, 이후 **GrabCut 알고리즘**이 이 정보를 사용하여 전경과 배경을 구분합니다.

### 2. GrabCut 알고리즘 실행
- `cv2.grabCut()`을 사용하여, 사용자가 지정한 **전경(Foreground)**과 **배경(Background)**을 기반으로 GrabCut 알고리즘이 동작합니다.
- 알고리즘은 **배경을 제거**하고 **전경을 추출**합니다.

### 3. 마우스를 드래그하며 연속적으로 영역 지정
- 마우스를 드래그하면서 **전경**과 **배경**을 연속적으로 지정할 수 있습니다. 
- 사용자는 여러 지점을 클릭하여 보다 **정교하게 전경과 배경을 구분**할 수 있습니다.

### 대화식 분할의 핵심
- **사용자 개입**: 사용자가 전경과 배경을 수동으로 지정함으로써, 자동화된 분할 알고리즘이 **보다 정확한 결과**를 얻을 수 있습니다.
- **GrabCut 알고리즘**: 사용자가 지정한 전경과 배경을 바탕으로 분할 작업을 수행하고, 추가적인 연산을 통해 **정교한 분할**을 제공합니다.

### 동작 과정
1. 사용자가 이미지에서 **전경(파란색)**과 **배경(빨간색)**을 클릭하여 지정합니다.
2. `cv2.grabCut()`이 **초기 사각형 영역**(rc)을 기준으로 배경과 전경을 분리합니다.
3. 사용자가 마우스로 클릭한 정보를 바탕으로 **GrabCut 알고리즘**이 동작하여, 전경과 배경을 더욱 **정교하게 구분**합니다.

### 전체 코드
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

### 결과 화면

초기 화면
![image](https://github.com/user-attachments/assets/fbfd0794-6533-4068-b0de-44e74c444372)

보이게할 부분 드래그
![image](https://github.com/user-attachments/assets/1743022f-9810-4d73-9131-479f942aaaac)

Enter 키 누른후 화면
![image](https://github.com/user-attachments/assets/e0f2bca0-6111-4827-b560-3e5642a46f77)

ESC 누른후 원본, 마스크, 배경제거 이미지 생성
![image](https://github.com/user-attachments/assets/d810b351-7ffa-442c-80de-fc8bc8195303)
