import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
import numpy as np

from tensorflow.keras import models, layers 

#현재 tensorflow가 즉시실행모드로 작동중 - 메모리를 어마어마하게 사용한다 
#이전 컴퓨터들은 작동안되는 경우가 있다ㅏ. 
#그럴경우 즉시 실행모드를 끄고 하자 
tf.compat.v1.disable_eager_execution()

(train_image, train_labels), (test_image, test_labels)=cifar10.load_data()
print( train_image.shape)
print( test_image.shape)
print( train_labels.shape)
print( test_labels.shape)

# import matplotlib.pyplot as plt 
# plt.imshow( train_image[0])
# plt.show()

def make_model(): #일반딥러닝
    network = models.Sequential()
    network.add( layers.Dense(64, activation='relu', input_shape=(32*32*3,)))
    network.add( layers.Dense(64, activation='relu'))
    network.add( layers.Dense(21, activation='relu'))
    network.add( layers.Dense(16, activation='relu'))
    network.add( layers.Dense(10, activation='softmax'))

    data = train_image.reshape( (train_image.shape[0], 32*32*3) )
    data = data/255

    network.compile( optimizer="rmsprop", loss="categorical_crossentropy", 
                     metrics=["accuracy"])

    from tensorflow.keras.utils import to_categorical 
    target  = to_categorical(train_labels)

    network.fit(data, target, epochs=50, batch_size=100)


def make_model2(): #일반딥러닝
    network = models.Sequential()

    network.add( layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)))
    network.add( layers.MaxPooling2D(2,2))
    network.add( layers.Conv2D(32, (3,3), activation='relu'))
    network.add( layers.MaxPooling2D(2,2))
    network.add( layers.Conv2D(32, (3,3), activation='relu'))
    network.add( layers.MaxPooling2D(2,2))
    network.add( layers.Flatten()) #CNN 과 완전연결층을 연결한다 
    network.add( layers.Dense(64, activation='relu'))
    network.add( layers.Dense(64, activation='relu'))
    network.add( layers.Dense(21, activation='relu'))
    network.add( layers.Dense(16, activation='relu'))
    network.add( layers.Dense(10, activation='softmax'))

    data = train_image/255

    network.compile( optimizer="rmsprop", loss="categorical_crossentropy", 
                     metrics=["accuracy"])

    from tensorflow.keras.utils import to_categorical 
    target  = to_categorical(train_labels)

    network.fit(data, target, epochs=20, batch_size=100)


if __name__ == "__main__":
    make_model()   
    make_model2()   
    
