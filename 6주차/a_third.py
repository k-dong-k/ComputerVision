import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import gc

# 1. 데이터 로드 및 라벨 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 이미지 크기 및 배치 사이즈 설정
IMG_SIZE = 128
BATCH_SIZE = 32

# 3. ImageDataGenerator로 실시간 이미지 리사이징 및 정규화
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow(
    tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)), y_train,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow(
    tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)), y_train,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# 4. 사전학습된 MobileNetV2 로드 (작고 빠름, Top 레이어 제거)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # 가중치 고정

# 5. 새로운 분류기 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 6. 컴파일 및 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🔄 전이학습 모델 훈련 시작")
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 7. 테스트 데이터 리사이즈 및 평가
x_test_resized = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE)) / 255.0
test_loss, test_acc = model.evaluate(x_test_resized, y_test)
print(f"✅ 전이학습(MobileNetV2) 모델 테스트 정확도: {test_acc:.4f}")

# 8. 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Transfer Learning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 9. 메모리 정리
tf.keras.backend.clear_session()
gc.collect()
