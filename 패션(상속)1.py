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
class MyModel(keras.Model):
    def __init__(self, output_cnt=10, activation_name="softmax"):
        #부모생성자를 반드시 호출해야 한다 
        super().__init__() #부모 생성자 호출 코드는 반드시 맨처음에 와야한다
        self.firstLayer = layers.Dense(64, activation="relu", name="firstLayer")
        self.secondLayer = layers.Dense(32, activation="relu", name="secondLayer")
        self.thirdLayer = layers.Dense(16, activation="relu", name="thirdLayer")
        #반환값 못보낸다 
        self.outputs = layers.Dense(output_cnt, activation=activation_name)

    def call(self, inputs):
        data = inputs["data"]#inputLayer
        
        features = self.firstLayer(data)
        features = self.secondLayer(features)
        features = self.thirdLayer(features)
        outputs  = self.outputs(features)
        return outputs 

model = MyModel()
outputs  = model({"data":X_train})
print( model.summary())

#컴파일 과정을 거친다.   최적화함수 :sgd, adam, rmsprop...
model.compile( optimizer="sgd", 
                 loss="categorical_crossentropy", #다중분류일때
                 metrics=['accuracy']) #평가항목, 정확도

#학습
hist = model.fit( {"data":X_train}, y_train, 
                    epochs=5, #학습횟수 
                    batch_size=100) #한번에 메모리에 불러오는 데이터양

train_loss, train_acc = model.evaluate({"data":X_train}, y_train)
print(f"훈련셋 손실 :  {train_loss}  정확도:{train_acc}")









