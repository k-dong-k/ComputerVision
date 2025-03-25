import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt

# 이미지를 skimage에서 가져오기
src = skimage.data.coffee()

# 초기사각형영역 (x, y, width, height)
rc = (50, 50, src.shape[1] - 100, src.shape[0] - 100)  # 이 값은 필요에 맞게 조정 가능

# 마스크 및 배경/전경 모델 초기화
mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iterCount = 1
mode = cv2.GC_INIT_WITH_RECT

# 초기 GrabCut 실행
cv2.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)
# 초기 이미지를 윈도우에 표시
mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]
cv2.imshow('dst', dst)  # 먼저 'dst' 윈도우를 표시
# 대화식 마스크 업데이트를 위한 마우스 이벤트 콜백 함수
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 클릭 - 전경
        cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)  # 전경은 파란색
        cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)  # 마스크에서 전경으로 설정
        cv2.imshow('dst', dst)
        
    elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 마우스 클릭 - 배경
        cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)  # 배경은 빨간색
        cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)  # 마스크에서 배경으로 설정
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
            

# 마우스 콜백 설정
cv2.setMouseCallback('dst', on_mouse)

# 배경 제거 및 최종 결과 업데이트
while True:
    key = cv2.waitKey()
    if key == 13:  # Enter 키로 GrabCut 실행
        # GrabCut을 마스크에 따라 실행하여 결과 업데이트
        cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 최종 전경/배경 마스크
        dst = src * mask2[:, :, np.newaxis]
        cv2.imshow('dst', dst)
    elif key == 27:  # Esc 키로 종료
        break

# 원본 이미지, 마스크 이미지, 배경 제거된 이미지를 matplotlib로 시각화
mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 이미지
axes[0].imshow(src)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 마스크 이미지 (배경은 0, 전경은 1)
axes[1].imshow(mask2, cmap='gray')
axes[1].set_title('Mask Image')
axes[1].axis('off')

# 배경이 제거된 이미지
axes[2].imshow(dst)
axes[2].set_title('Foreground Removed')
axes[2].axis('off')

# 이미지 표시
plt.show()

cv2.destroyAllWindows()
