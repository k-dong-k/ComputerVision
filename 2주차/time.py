import cv2 as cv
import numpy as np
import time
import psutil

def memory_usage(message: str = 'debug'):  #메모리 사용량 체크 하무
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss : 10.5f} MB")
    
def my_cvtGray1(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r,c] = 0.114 * bgr_img[r, c, 0] + 0.587 * bgr_img[r, c, 1] + 0.299 * bgr_img[r, c, 2]
    return np.uint8(g)

def my_cvtGray2(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    g = 0.114 * bgr_img[:, :, 0] + 0.587 * bgr_img[:, :, 1] + 0.299 * bgr_img[:, :, 2]
    return np.uint8(g)
memory_usage('#0') # 실행 전
img = cv.imread('soccer.jpg')

# cp = (img.shape[1] / 2, img.shape[0] / 2) # 영상의 가로 1/2, 세로 1/2
rot = cv.getRotationMatrix2D(:, 45, 1.5) # 20도 회전, 스케일 0.5배
cv.imshow('img', img)

cv.waitKey()
cv.destroyAllWindows()

memory_usage('#1') # 이미지 메모리 체크

start = time.time()
my_cvtGray1(img)
print('1st function:', time.time() - start)
memory_usage('#2')

start = time.time()
my_cvtGray2(img)
print('2nd function:', time.time() - start)
memory_usage('#3')

start = time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('OpenCV function:', time.time() - start)
memory_usage('#4')