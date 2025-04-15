import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리: 정규화 및 평탄화
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 3. 신경망 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')  # 숫자 0~9 분류
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련 (History 객체로 결과 저장)
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 6. 테스트 세트로 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# 7. 성능 시각화
plt.figure(figsize=(12, 5))

plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# 정확도 시각화
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('정확도')
plt.legend()

plt.tight_layout()
plt.show()
