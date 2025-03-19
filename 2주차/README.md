# 컴퓨터 비전

컴퓨터 비전 2주차 실습 과제 1번


![image](https://github.com/user-attachments/assets/a05c3def-f463-439b-8880-0da0c9ef824a)


cv.imread()를 사용하여 이미지 불러옴

     img = cv.imread('soccer.jpg')

Cv.cvtColor()를 사용하여 그레이스케일로 변환

     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 


Cv.threshold()를 사용하여 이진화,  Cv.calcHist()를 사용하여 히스토그램을계산, matplotlib로시각화

     t,bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
     h = cv.calcHist([bin_img],[0],None,[256],[0,256])


전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

gh = cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(gh,color = 'b', linewidth = 1)
plt.show()

t,bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

h = cv.calcHist([bin_img],[0],None,[256],[0,256])
cv.imshow('R channel binarization', bin_img)
plt.plot(h,color = 'r', linewidth = 1)
plt.show()
```

결과 화면

그레이 히스토그램

![image](https://github.com/user-attachments/assets/913dc711-f561-4bf7-a29e-0b25664d8fa9)



이진화의 경우 0과1로만 나오기떄문에 히스토그램이 아래와 같은 형태로 나옴

![image](https://github.com/user-attachments/assets/99e102a7-6680-47cf-ab07-080794a28920)



---

컴퓨터 비전 2주차 실습 과제 2번

![image](https://github.com/user-attachments/assets/ad205262-0358-4dbb-97dc-eb09d88fe0aa)


cv.getStructuringElement()를 사용하여 사각형 커널(5x5) 만듬

     se = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

cv.morphologyEx()를 사용하여 각 모폴로지연산을 적용
```python
dilating = cv.morphologyEx(b, cv.MORPH_DILATE, se)
eroding = cv.morphologyEx(b, cv.MORPH_ERODE, se)
opening = cv.morphologyEx(b, cv.MORPH_OPEN, se)
closing = cv.morphologyEx(b, cv.MORPH_CLOSE, se)
```

전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

t,bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]
plt.imshow(b, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

se = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

dilating = cv.morphologyEx(b, cv.MORPH_DILATE, se)

eroding = cv.morphologyEx(b, cv.MORPH_ERODE, se)

opening = cv.morphologyEx(b, cv.MORPH_OPEN, se)

closing = cv.morphologyEx(b, cv.MORPH_CLOSE, se)
merge = np.hstack((dilating, eroding, opening, closing))

cv.imshow('d,e,o,c',merge)
cv.waitKey(0)
cv.destroyAllWindows()
```

결과 화면

팽창, 침식, 열림, 닫힘 순으로 나옴
![image](https://github.com/user-attachments/assets/b5e9053f-36f6-4e2c-8770-568863bc67af)


---

컴퓨터 비전 1주차 실습 과제 3번

![image](https://github.com/user-attachments/assets/74e0a705-242d-4887-a3d3-0633f9785242)


cv.getRotationMatrix2D()를 사용하여 회전 변환 행렬을생성
```python
row, col = img.shape[:2]
cp = (col / 2, row / 2) 
rot = cv.getRotationMatrix2D(cp, 45, 1.5)
```
cv.warpAffine()를 사용하여 이미지를 회전 및 확대, cv.INTER_LINEAR을 사용하여 선형보간을 적용

     dst = cv.warpAffine(img, rot, (int(col * 1.5), int(row * 1.5)), flags = cv.INTER_LINEAR)

원본이미지와 회전 및 확대 된 이미지를 한 화면에 비교 (비교할때 높이를 맞춰야해서 크기조정)
```python
if img.shape[0] != dst.shape[0]:
    dst = cv.resize(dst, (int(col * 1.5), img.shape[0]))
  
result = np.hstack([img, dst])
```
결과화면

![image](https://github.com/user-attachments/assets/1050d58d-e539-4655-871f-e5fcd216c040)
