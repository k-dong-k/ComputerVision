# 1. 간단한 이미지 분류기 구현

📢 설명

* 손글씨 숫자이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현

📖 요구사항

* MNIST 데이터셋을 로드

* 데이터를 훈련 세트와 테스트 세트로 분할

* 간단한 신경망 모델을 구축

* 모델을 훈련시키고 정확도를 평가


### MNIST 데이터셋 로드, 데이터를 훈련 세트와 테스트 세트로 분할

      (x_train, y_train), (x_test, y_test) = mnist.load_data()

* mnist.load_data()는 숫자 이미지 데이터 가져오는 함수

* x_train, y_train: 훈련용 데이터 (이미지와 정답 숫자)

* x_test, y_test: 테스트용 데이터 (모델이 잘 학습했는지 확인할 때 사용)

ex) x_train[0]은 숫자 이미지 하나 (예: 28x28 크기), y_train[0]은 이미지가 어떤 숫자인지 (예: 5)


### 간단한 신경망 모델을 구축
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```
Sequential 

* 신경망 모델을 순서대로 층을 쌓아서 만든다는 뜻

Dense

* 128 -> 128개 뉴런 층

* activation = 'relu' -> 활성화 함수 ReLU 사용

* input_shape=(784,) -> 입력 데이터는 784개 숫자 (28x28 이미지 한장 을 뜻함)

* 10 -> 출력이 10개라는 뜻 (숫자 0~9 총 10개 중 하나로 분류)

* activation = 'softmax' -> 확률처럼 만들어서 가장 높은 값을 정답으로 선택


### 모델을 훈련시키고 정확도를 평가
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")
```
* loss='sparse_categorical_crossentropy -> 정답과 예측값이 얼마나 다른지 계산해주는 방법

* metrics = ['accuracy'] -> 모델의 성능을 측정하는 기준을 정할 때 사용 (accuracy - 정확도 측정)

* model.evaluate(x_test, y_test) -> 테스트 데이터를 넣어서 얼마나 잘 맞추는지 평가


### 전체 코드
```python
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

```


### 실행 결과

![image](https://github.com/user-attachments/assets/ff5c3346-0c51-4992-b96a-e3fa51867e4b)



# 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

📢 설명

*  CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행

📖 요구사항

* CIFAR-10 데이터셋을 로드

* 데이터전처리(정규화등)를 수행

* CNN 모델을 설계하고 훈련

* 모델의 성능을 평가하고, 테스트 이미지에 대한 예측을 수행


### CIFAR-10 데이터셋을 로드
```python
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
```
* CIFAR-10은 32x32 크기의 컬러 이미지로 구성된 데이터셋


### 데이터 전처리를 수행
```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```
* 이미지 픽셀 값 (0~255) -> 0 ~ 1 사이 소수로 변경(학습을 쉽게 하기 위해)


### CNN 모델을 설계하고 훈련
```python
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))
```
Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))

* 32 : 사진의 크기가 32 x 32 이므로 32개의 커널을 만듬

* (3, 3) : 각 필터의 크기 3 픽셀 -> 작은 부분을 보며 특징을 찾아냄

* activation = 'relu' : 음수는 0으로 양수는 그대로 통과시킴 -> 복잡한 패턴 잘 배움

* input_shape = (32, 32, 3) : CIFAR-10 한 이미지의 크기 (3 -> RGB의미)

MaxPooling2D((2, 2))

* (2, 2) : 2픽셀 × 2픽셀 -> 이미지 크기를 줄이면서 중요한 특징만 남김 + 과적합 방지

  
* loss='sparse_categorical_crossentropy': 숫자(정수)로 되어 있을 때 쓰는 분류용 손실함수


### 모델의 성능을 평가, 테스트 이미지에 대한 예측을 수행
```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n 테스트 정확도: {test_acc:.4f}')

predictions = model.predict(x_test)
```

### 전체 코드
```python
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
print(f'\n 테스트 정확도: {test_acc:.4f}')

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
```

### 결과 화면

![image](https://github.com/user-attachments/assets/9366cfba-a81d-4c64-9adf-2904897b9f3f)

