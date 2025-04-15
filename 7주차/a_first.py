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
