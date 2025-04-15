# 1. SORT 알고리즘을 활용한 다중 객체추적기 구현

📢 설명

* 이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중객체를 실시간으로 추적하는 프로그램을 구현합니다.

* 이를통해 객체추적의 기본개념과 SORT 알고리즘의 적용방법을 학습할 수 있습니다.

📖 요구사항

* 객체 검출기 구현 : YOLOv4와 같은 사전 훈련 된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출합니다.

* mathworks.com SORT 추적기 초기화 : 검출된 객체의 경계상자를 입력으로 받아 SORT 추적기를 초기화합니다.

* 객체추적 : 각 프레임마다 검출된 객체와 기존 추적객체를 연관시켜 추적을 유지합니다.

* 결과시각화 : 추적된 각 객체에 고유ID를 부여하고, 해당ID와 경계상자를 비디오프레임에 표시하여 실시간으로 출력합니다.


### 객체 검출기 구현
```python
def construct_yolo_v4(cfg_path='yolov4.cfg', weights_path='yolov4.weights', names_path='coco_names.txt'):
    # 클래스 이름 로딩
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # YOLOv4 모델 불러오기
    model = cv.dnn.readNet(weights_path, cfg_path)

    # 출력 레이어 추출
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    
    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = img.shape[:2]
    
    # YOLO 입력 포맷(blob) 생성
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)

    # YOLO 모델 추론 실행
    outputs = yolo_model.forward(out_layers)

    boxes, confidences, class_ids = [], [], []

    # 추론된 결과 해석
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, x + w, y + h])  # 좌상단(x,y) ~ 우하단(x+w, y+h)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제(NMS) 적용하여 중복 제거
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result = [boxes[i] + [confidences[i]] + [class_ids[i]] for i in range(len(boxes)) if i in indices]
    return result

```

* YOLOv4 모델과 관련된 설정 파일(.cfg), 가중치 파일(.weights), 클래스 이름 파일(.txt)을 로딩하여 모델을 초기화

* model: YOLOv4 모델 객체

* out_layers: 출력 레이어 이름 리스트

* class_names: 클래스 이름 리스트 (예: person, car 등)

* Confidence가 높은 박스만 필터링하고, 중복된 박스는 제거(NMS).

* 최종적으로 [x1, y1, x2, y2, confidence, class_id] 형식의 리스트 반환.


### SORT 추적기 초기화
```python
from sort.sort import Sort

# 추적기 객체 생성
sort = Sort()
```

* github에서 sort 파일을 다운받고 초기화함


### 객체 추적
```python
# 객체 검출
detections = yolo_detect(frame, model, out_layers)

# 사람만 추출 (COCO 클래스에서 사람 class_id == 0)
persons = [det for det in detections if det[5] == 0]

# 추적기 업데이트
if persons:
    tracks = sort.update(np.array(persons))  # 형식: [x1, y1, x2, y2, track_id]
else:
    tracks = sort.update()  # 추적 유지
```

* 탐지 결과 중 사람만 추출하여 SORT에 전달 (class_id == 0은 COCO에서 "person")

* sort.update()를 통해 새로운 바운딩 박스에 ID를 연결

* track_id는 SORT가 부여하는 고유한 추적 ID로, 같은 사람을 여러 프레임 동안 계속 추적

### 결과 시각화
```python
for track in tracks:
    x1, y1, x2, y2, track_id = track.astype(int)
    color = colors[track_id % 100]  # ID마다 색 다르게
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
```

* 각 객체의 바운딩 박스를 색상으로 그림

* 객체 ID를 텍스트로 표시하여 추적이 잘 되고 있는지 확인할 수 있게 함

* track_id % 100을 통해 최대 100개의 색상을 순환하여 각 객체를 구분


### 전체코드
```python
import numpy as np
import cv2 as cv
import sys
from sort.sort import Sort

def construct_yolo_v4():
    # COCO 클래스 이름 로딩
    with open('coco_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # YOLOv4 모델 로딩
    model = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    
    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[:2]
    
    # YOLOv4는 보통 416x416 입력을 사용
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    outputs = yolo_model.forward(out_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = [boxes[i] + [confidences[i]] + [class_ids[i]] for i in range(len(boxes)) if i in indices]
    return result

# 모델 초기화
model, out_layers, class_names = construct_yolo_v4()
colors = np.random.uniform(0, 255, size=(100, 3))
sort = Sort()

# 웹캠 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득 실패. 루프 종료.')

    # 객체 검출
    detections = yolo_detect(frame, model, out_layers)
    # 사람 클래스만 필터링 (COCO에서 0번 class가 "person")
    persons = [det for det in detections if det[5] == 0]

    # 객체 추적 업데이트
    if persons:
        tracks = sort.update(np.array(persons))
    else:
        tracks = sort.update()

    # 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        color = colors[track_id % 100]
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 출력
    cv.imshow('YOLOv4 + SORT Tracking', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```

### 실행 결과


# 2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

📢 설명

* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출, 실시간 영상에 시각화하는 프로그램을 구현

📖 요구사항

* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴랜드마크 검출기를 초기화

* OpenCV를 사용하여 웹캠으로부터 실시간영상을 캡처

* 검출된 얼굴랜드마크를 실시간 영상에 점으로 표시

* ESC 키를누르면 프로그램이 종료되도록 설정


### Mediapipe의 FaceMesh 모듈을 사용하여 얼굴랜드마크 검출기를 초기화
```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
* mediapipe.solutions.face_mesh: Mediapipe에서 FaceMesh 기능을 가져오기 위한 모듈

* static_image_mode=False: 실시간 영상이므로 각 프레임에서 추적을 유지

* max_num_faces=1: 한 번에 추적할 얼굴은 1명으로 제한

* min_detection_confidence=0.5: 얼굴이 검출되었다고 판단하기 위한 최소 확신 값

* min_tracking_confidence=0.5: 추적을 계속하기 위한 최소 확신 값.

### OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처
```python
import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득 실패")
        break

```
* cv.VideoCapture(0): 디폴트 웹캠을 열어 영상 스트림을 가져옴

* cv.CAP_DSHOW: 윈도우에서 DirectShow 방식으로 연결

* .read(): 현재 프레임을 캡처 (성공 여부는 ret, 프레임 자체는 frame)

* if not ret: 프레임을 제대로 못 가져왔을 경우 루프를 종료

### 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시
```python
rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
result = face_mesh.process(rgb_frame)

if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
```
* Mediapipe는 RGB 이미지를 입력으로 받기 때문에 OpenCV의 BGR 이미지를 변환

* face_mesh.process(...): 현재 프레임에서 얼굴을 감지하고, 랜드마크를 반환

* result.multi_face_landmarks: 얼굴이 검출되었는지 확인

* landmark.x, landmark.y: 각 포인트는 비율(0~1) 형태로 반환되므로, 화면 크기에 맞게 곱해줌

* cv.circle(...): 각 랜드마크를 초록색 점으로 프레임에 표시

### ESC 키를 누르면 프로그램이 종료되도록 설정
```python
cv.imshow('MediaPipe FaceMesh', cv.flip(frame, 1))

if cv.waitKey(5) & 0xFF == 27:
    break

cap.release()
cv.destroyAllWindows()
```
* & 0xFF == 27: 입력된 키의 ASCII 코드가 27(Esc)일 경우 종료

### 전체 코드
```python
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
```

### 결과 화면








