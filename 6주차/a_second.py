import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 2. 데이터 전처리 - 정규화 (0~255 -> 0~1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. CNN 모델 설계
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # CIFAR-10은 10개의 클래스
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# 6. 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n✅ 테스트 정확도: {test_acc:.4f}')

# 7. 테스트 이미지 예측
predictions = model.predict(x_test)

# 8. 예측 결과 시각화
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_image(i, predictions_array, true_label, img):
    true_label, img = int(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array[i])
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions_array[i]):.2f}%)\n[정답: {class_names[true_label]}]",
               color=color)

# 시각화
plt.figure(figsize=(6,3))
plot_image(0, predictions, y_test, x_test)
plt.show()
