import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # OpenCV는 BGR, Matplotlib은 RGB 형식

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)  # Tlow=100, Thigh=200 설정

lines = cv.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=5)

results = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(results, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 빨간색 (R=255, G=0, B=0), 두께=2



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
