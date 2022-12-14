import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers
tf.random.set_seed(1234)

import os 
SEED=1  # 시드 고정
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(1234)
# 1. 데이터셋 불러오기(훈련셋이 60000, 테스트셋이 10000)
(X_train, y_train), (X_test,  y_test) = fashion_mnist.load_data()
print("훈련셋 : ", X_train.shape, y_train.shape)
# print("훈련셋 : ", y_train.shape)
print("검증셋(test) : ", X_test.shape, y_test.shape)
# print("검증셋 : ", y_test.shape)

import matplotlib.pyplot as plt
def showImage(index):
    plt.imshow(X_train[0])
    plt.show()

# showImage(1)

def showImage2(row, col):
    plt.figure(figsize=(10, 5))
    for i in range(row*col):
        plt.subplot(row, col, i+1) # subplot은 (행, 열) 구역번호
        plt.imshow(X_train[i])

    plt.show()

# 이미지 저장하기
# showImage2(5,10)
# import PIL.Image as pilimg

# def saveImage(cnt):
#     for i in range(0, cnt):
#         pilimg.fromarray(X_train[i]).save(f"./images/image{i}.png")

# saveImage(1000)

# 딥러닝 - 심층신경망 신경세포1 -> 신경세포2 -> 신경세포3

# 1.딥러닝도 입력데이터는 2차원 numpy 이어야한다.
# 2.결과(label) 데이터는 1차원, 원핫인코딩(2차원) 둘 중 하나 선택

# feature의 개수가 28 * 28개 # 딥러닝 중요
X_train = X_train.reshape( (X_train.shape[0], 28 * 28) )
X_test = X_test.reshape( (X_test.shape[0], 28 * 28) )

# 정규화( 0~1사이에) # 이미지의 경우
X_train = X_train/255 # .점은 플롯
X_test = X_test/255

# 출력데이터 (원핫 인코딩)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical (y_train)
y_test = to_categorical (y_test)

# 네트워크 또는 모델을 만든다.
network = models.Sequential()

"""
# layers.Dense 심층 신경망 함수를 이용해 신경망을 쌓아간다.
# 첫번째 layers.에는 input_shape=(feature의 개수)
# activation 함수 sigmoid, relu(성능이 좋다) 학자들이 좋다해서 relu를 고정해서 사용

layers 개수 상관없이 결과에 따른 높은 확률에 맞춰 사용한다.

# 맨 마지먹 layer는 회귀냐 이진분류나 다중분류에 따라 방식이 달라진다.
# fashon_minst 라벨이 10개 라벨의 개수, 다중분류ㅠ일때는 softmax 함수를 사용한다.
softmax 함수는 결과값들을 0~1에 이르는 확률로 만들어주는 함수이다.
"""
# 첫번째 층에 있는 숫자 32는 피처의 개수가 32개로 줄어서 나온다.
network.add(layers.Dense(64, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))


# 컴파일 과정을 거친다. 최적화 함수 : sgd, adam, rmsprop...
network.compile( optimizer='sgd', # optimizer 최적화 함수
                 loss='categorical_crossentropy', # 손실함수 - 다중분류일때 고정으로 사용한다
                 metrics=['acc'])            # 평가항목, 정확도

# 학습
hist = network.fit( X_train, y_train, 
                    epochs=100, # 학습횟수
                    batch_size=100) # 한번에 메모리 불러오는 데이터의 양

train_loss, train_acc = network.evaluate(X_train, y_train)
test_loss, test_acc = network.evaluate(X_test, y_test)

print(f"훈련셋 손실 : {train_loss} 정확도:{train_acc}")
print(f"테스트셋 손실 : {test_loss} 정확도:{test_acc}")

