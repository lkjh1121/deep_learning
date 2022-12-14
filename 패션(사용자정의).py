import tensorflow as tf 
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras import models, layers 
import os 
SEED=1
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

#1.데이터셋 불러오기(훈련셋이 60000, 테스트셋이 10000)
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape) 
# (60000, 28, 28)  이미지 크기 : 28 by 28 색상정보 없음
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import matplotlib.pyplot as plt 
def showImage(index):
    plt.imshow(X_train[index])
    plt.show()

#showImage(1)

def showImage2(row, col):
    plt.figure(figsize=(10, 5))
    for i in range(row*col):
        plt.subplot(row, col, i+1) #행, 열, 구역번호
        plt.imshow( X_train[i])
    
    plt.show()

#showImage2(5, 10)

#이미지 저장해보기
import PIL.Image as pilimg
def saveImage(cnt):
    for i in range(0, cnt):
        pilimg.fromarray(X_train[i]).save(f"./images/image{i}.png")

#saveImage(1000)

#딥러닝 - 심층신경망  신경세포1 -> 신경세포2 -> 신경세포3

#1.딥러닝도 입력데이터는 2차원 numpy 이어야 한다 
#2.결과(label)데이터는 1차원, 원핫인코딩(2차원)둘중 하나 선택 
#feature의 개수가 28 * 28 개 
X_train = X_train.reshape((X_train.shape[0], 28 * 28) )  
X_test = X_test.reshape( (X_test.shape[0], 28*28) )

#정규화(0~1사이에) 
X_train = X_train/255   #
X_test = X_test/255

#원핫인코딩 
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow import keras  

# 콜백 함수 - 사용자 정의 함수로 일부 내용을 대체 할 수 있다.
# 함수는 내가 만들지만 호출은 keras에서 호출한다.

class MyCalculator: # 케라스에서 만든것
    def __init__(self, x=10, y=20):
        self.x = x
        self.y = y

    def call(self, calculator):
        #이 함수가 매개변수 함수를 전달받는다. 이 함수는 매개변수가 x, y 두개를 갖는다
        print(f"x= {self.x} y = {self.y} result={calculator(self.x, self.y)}")

# 본인이 호출
def add(x,y):
    return x+y

m1 = MyCalculator(100, 200)
m1.call( add )


# 모델 만들기
def get_model():
    inputs = keras.Input( shape=(28*28,))
    features = layers.Dense(512, activation='relu')(inputs)
    features = layers.Dense(256, activation='relu')(features)
    features = layers.Dense(128, activation='relu')(features)
    features = layers.Dense(64, activation='relu')(features)
    outputs = features = layers.Dense(10, activation='softmax')(features)
    model = keras.Model(inputs, outputs)
    return model

model = get_model()
model.compile( optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")


callback_list = [
    # 합리적인 val_loss, val_accurcy 일때 모델을 파일로 저장해라
    keras.callbacks.ModelCheckpoint(
        filepath="fashion_model.keras", 
        monitor="val_closs",    # 검증 데이터의 손실값을 모니터링 해라 
        save_best_only=True     # 모니터링 되고 있는 값중에 가장 좋은 값일때 저장해라
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2 #  바로 끝내지 말고 2번 더 돌려라
    )]

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=100, batch_size=100, callbacks=callback_list)




