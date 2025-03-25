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

