"""
이미 학습된 데이터의 특성을 추출해서 사용해보자 
VGG16 -> VGG19  , Resnet 

CNN - 합성곱의 특징 - 분류가 아니고, 특성추출 
완전연결망 - 분류 

실제데이터가 작아야(이미지가 별로 없을때 학습효과가 너무 낮다)
VGG19, Resnet 수십만장의 이미지를 가지고 학습을 했음 
충분히 데이터가 많다면 안써도 된다. 
VGG19, 16 이던 케라스가 갖고 있다
"""

from tensorflow import keras 
from keras.applications.vgg19 import VGG19

conv_base = VGG19( 
    weights="imagenet",
    include_top=False, #top - 완전연결망은 배제하고  
    input_shape=(180, 180, 3) #이미지크기 - 180은 내마음, 데이터셋과 크기가 맞아야 한다 
)

conv_base.summary() 

"""
1.특성추출
  1-1.  데이터 증식을 하지 않는 경우 : 단순히 특성만 추출한다  
  1-2.  데이터 증식을 하는 경우 

2.미세조정 

"""


from keras.utils import image_dataset_from_directory
import pathlib
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet import ResNet
from keras.applications.resnet import preprocess_input

def createDataset(base_dir = "./dogs-vs-cats/cats_vs_dogs_small"):
    base_dir = pathlib.Path(base_dir)
    train_ds = image_dataset_from_directory( 
        base_dir/"train", 
        image_size=(180,180),
        batch_size=16)

    test_ds = image_dataset_from_directory( 
        base_dir/"test", 
        image_size=(180,180),
        batch_size=16)

    valid_ds = image_dataset_from_directory( 
        base_dir/"validation", 
        image_size=(180,180),
        batch_size=16)

    return train_ds, test_ds, valid_ds  


"""
conv_base : 특성추출 
이미지를 preprocess_input(이미지벡터) 거쳐서 전처리 작업하고 
predict(전처리된이미지) - 특성추출
우리가 가져온 이미지 -> convnet 을 통과시켜서 특성만 가져오자 
"""
import matplotlib.pyplot as plt
import numpy as np 

#1.데이터셋 먼저 만들고
train_ds, test_ds, valid_ds = createDataset()
#2.특성을 추출하고 

#3.모델을 만들어서 학습하자 
#conv_base.summary() 맨상위층이 5,5,512
conv_base.trainable = False #(합성곱 기반층을 동결시킨다) 

# 미세조정은 컨브넷의 일부는 동결시키고 일부는 해제하기
conv_base.trainable=True
for layer in conv_base.layers[:-4]:
    print("동결 : ", layer.name)
    layer.trainable=False
conv_base.summary()

# #증식에 필요한 정보

data_argumentation = keras.Sequential([
    layers.RandomFlip("horizontal") ,
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.2) 
])

inputs = keras.Input(shape=(180,180,3))
from keras import layers

x = data_argumentation()(inputs)
x = preprocess_input(x) #전처리를 하고
x = conv_base(x) 
x = layers.Flatten()(x)  
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x) #절반의 데이터를 버린다. 과대적합을 막으려고 
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model( inputs, outputs) #모델만들기 

# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 미세조정시 
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(learning_rate=1e-5), metrics=["accuracy"])

model.fit( 
    train_ds.repeat(5), 
    epochs=5, 
    validation_data = valid_ds
)


