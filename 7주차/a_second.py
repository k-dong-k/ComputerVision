import cv2 as cv
import mediapipe as mp

# MediaPipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠에서 영상 캡처
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득 실패")
        break

    # RGB로 변환 후 얼굴 랜드마크 처리
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # 얼굴 랜드마크가 검출되었을 경우
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 각 랜드마크를 점으로 표시
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])  # 이미지 너비 기준 x좌표
                y = int(landmark.y * frame.shape[0])  # 이미지 높이 기준 y좌표
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 점 그리기 (초록색)

    # 좌우 반전 후 출력
    cv.imshow('MediaPipe FaceMesh', cv.flip(frame, 1))

    # ESC 키를 누르면 종료 (ASCII 코드 27)
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
