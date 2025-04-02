# 1. SIFT를 이용한 특징점 검출 및 시각화


📢 설명

* 주어진이미지(mot_color70.jpg)를이용하여SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을 검출하고 이를 시각화하세요


📖 요구사항

* cv.imread()를 사용하여 이미지를 불러옵니다.
  
* cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다.

* detectAndCompute()를 사용하여 특징점을 검출합니다.
  
* cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화합니다.
  
* matplotlib을 이용하여 원본이미지와 특징점이 시각화된 이미지를 나란히 출력하세요



### 이미지 불러오기

```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mot_color70.jpg')
```
* cv.imread() 함수는 지정된 파일 경로에서 이미지를 읽어옴
* img는 불러온 이미지 데이터를 저장하는 변수


### SIFT 객체 생성

    sift = cv.SIFT_create()

* cv.SIFT_create() 함수는 SIFT 알고리즘을 사용해 특징점을 추출하기 위핸 객체 생성


### 특징점 검출 및 기술자 계산
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kp, des = sift.detectAndCompute(gray, None)
```
* cv.cvtColor() 함수는 이미지를 BGR에서 그레이스케일로 변환 (SIFT 알고리즘은 색상 정보보다 형태 정보를 주로 사용하기 때문에 그레이스케일로 변환하는 것이 일반적)
* sift.detectAndCompute() 함수는 입력 이미지에서 특징점을 검출하고, 각 특징점에 대한 기술자를 계산


### 특정점 시각화

    gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


* cv.drawKeypoints() 함수는 검출된 특징점을 이미지에 그려서 시각화
* flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS는 특징점을 풍부한 정보를 포함한 형태로 그리기 위한 옵션 -> 특징점의 크기와 방향 등도 함께 표시


### 이미지 출력
```python
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('SIFT Image')
plt.axis('off')

plt.show()
```


### 전체 코드

```python
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
```


### 실행 결과


![image](https://github.com/user-attachments/assets/786fdb72-d772-4c6e-bf08-df4edfcef560)




# 2. SIFT를 이용한두영상간특징점매칭


📢 설명

* 두개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화하세요.


📖 요구사항

* cv.imread()를 사용하여 두개의 이미지를 불러옵니다.
  
* cv.SIFT_create()를 사용하여 특징점을 추출합니다.
  
* cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭합니다.
  
* cv.drawMatches()를 사용하여 매칭결과를 시각화합니다.
  
* matplotlib을 이용하여 매칭결과를 출력하세요


### cv.imread()를 사용하여 두개의 이미지를 불러오기

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 

img2 = cv.imread('mot_color83.jpg') 
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
```

* cv.imread()를 사용하여 이미지 읽어옴, 첫번쨰 이미지 img1은 특정영역을 슬라이싱함

### SIFT 객체 생성 및 특징점 추출
```python
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)  

kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수: ', len(kp1), len(kp2))
```
* cv.SIFT_create()를 사용하여 SIFT 객체를 생성
* detectAndCompute()를 사용하여 두 이미지에서 특징점(kp1, kp2)과 그에 해당하는 기술자(des1, des2)를 추출

### cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
```python
flann_matcher = cv.FlannBasedMatcher() 
knn_match = flann_matcher.knnMatch(des1, des2, 2)
```

### 매칭 결과 시각화
```python
img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

### 전체 코드
```python
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')[190:350,440:560]
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('특징점 개수 : ', len(kp1), len(kp2))

start = time.time()
flann_matcher = cv.FlannBasedMatcher()
knn_match = flann_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance) < T :
        good_match.append(nearest1)
print('매칭에 걸린 시간 : ', time.time() - start)

img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype = np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FLANN-Based Feature Matching')
plt.show()
```

### 실행 결과

![image](https://github.com/user-attachments/assets/e4cbcc50-d28e-4c56-94b8-143bfc98e9e7)



# 3. 호모그래피를이용한이미지정합(Image Alignment)

📢 설명

* SIFT 특징점을 사용하여 두 이미지간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬하세요.
* 샘플파일로img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택하세요.

📖 요구사항

* cv.imread()를 사용하여 두개의 이미지를 불러옵니다.

* cv.SIFT_create()를 사용하여 특징점을 검출합니다.
  
* cv.BFMatcher()를 사용하여 특징점을 매칭합니다.
  
* cv.findHomography()를 사용하여 호모그래피 행렬을 계산합니다.
  
* cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른이미지와 정렬합니다.
  
* 변환된이미지를 원본이미지와비교하여출력하세요.


###  cv.BFMatcher()를 사용하여 특징점을 매칭
```python
bf_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
knn_match = bf_matcher.knnMatch(des1, des2, 2)  # 최근접 2개
```
* cv.BFMatcher()는 특징점들을 매칭하기 위한 객체, cv.NORM_L2는 L2 거리(유클리디언 거리)를 사용하여 매칭을 수행
* knnMatch()는 각 특징점에 대해 최근접 2개의 매칭을 찾고 매칭 결과는 knn_match에 저장

### cv.findHomography()를 사용하여 호모그래피 행렬을 계산

    H, _ = cv.findHomography(points1, points2, cv.RANSAC)

* cv.findHomography()는 두 점 집합 사이의 호모그래피 행렬을 계산
* points1과 points2는 매칭된 특징점들의 좌표, 이 좌표들을 기반으로 호모그래피 행렬을 추정
* cv.RANSAC은 노이즈나 잘못된 매칭을 제외하기 위해 RANSAC 알고리즘을 사용

### cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬

    img1_warped = cv.warpPerspective(img1, H, (w2, h2))

* cv.warpPerspective()는 이미지를 호모그래피 행렬 H를 사용하여 변환
* H는 이미지 1을 이미지 2의 좌표계에 맞추기 위한 변환 행렬
* (w2, h2)는 이미지 2의 크기를 설정하여, 변환된 이미지가 적절히 배치되도록 함

  
### 변환된 이미지를 원본 이미지와 비교하여 출력
```python
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
```

* 첫 번째 이미지는 이미지 1
* 두 번째 이미지는 호모그래피 변환이 적용된 이미지
* 세 번째 이미지는 이미지 2


### 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('img1.jpg')
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

alpha = 0.5  # 투명도 조절 (0.5면 반반 섞임)
blended = cv.addWeighted(img1_warped, alpha, img2, 1 - alpha, 0)

plt.figure(figsize=(12, 6))
    
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.title("Image 1")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img1_warped, cv.COLOR_BGR2RGB))
plt.title("Homography Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.title("Image 2")
plt.axis('off')

plt.show()

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(blended, cv.COLOR_BGR2RGB))
plt.title("Blended Image")
plt.axis('off')
plt.show()

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("Feature Matches")
plt.axis('off')
plt.show()

```

### 실행결과

![image](https://github.com/user-attachments/assets/afb75c79-a947-4ab6-8c40-aca7134ed3a1)



![image](https://github.com/user-attachments/assets/daed1897-da82-4152-8849-ccc75c757232)


![image](https://github.com/user-attachments/assets/6030eae9-0e06-4453-9a59-6f9fa9ea14a9)
