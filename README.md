# 컴퓨터 비전

컴퓨터 비전 1주차 실습 과제

![image](https://github.com/user-attachments/assets/e5581abb-b905-458d-8495-ae00082d38c8)

cv.cvtColor() 함수를 사용해 이미지를 그레이 스케일로 변환 후 저장

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환
    cv.imwrite('soccer_gray.jpg', gray)  # 영상을 파일에 저장

원본 이미지와 그레이 스케일 이미지를 가로로 출력하기위해 채널수 맞춘 후 연결

    gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    merged = np.hstack((img, gray_3ch))
    
