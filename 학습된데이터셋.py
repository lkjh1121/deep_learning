# pip install gdown 
# import gdown
# gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')
#https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

from tensorflow.keras.utils import image_dataset_from_directory 
#디렉토리로 부터 데이터를 불러서 사용한다 

import os, shutil, pathlib 

#train폴더에 있는 dog와 cat을 각각 폴더만들고 파일을 찾아서 옮기도록 하자 
base_dir = pathlib.Path("./Garbageclassification")
new_base_dir = pathlib.Path("./Garbageclassification_data")

def make_subset(subset_name, train_size=0.7):
    for category in ("cardboard", "glass", "metal", "paper", "plastic", "trash"):
        dir = new_base_dir/subset_name/category
        os.makedirs(dir) #경로가 없으면 만든다. 
        fnames = os.listdir(base_dir/category)
        if subset_name=="train":
            start_index=0
            end_index = int(len(fnames)*0.7)
        else:
            start_index=int(len(fnames)*0.7)
            end_index = len(fnames)
        for i in range(start_index, end_index):
            fname = fnames[i]
            shutil.copyfile( src=base_dir/category/fname, dst=dir/fname)

#make_subset("train",  train_size=0.7)
#make_subset("test", train_size=0.7)

train_dataset = image_dataset_from_directory( 
    new_base_dir/"train",
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=(180, 180), 
    batch_size=16 ,
    label_mode='categorical'    ########## 추가, default-int           
)

validation_dataset = image_dataset_from_directory( 
    new_base_dir/"train",
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(180, 180), 
    batch_size=16,
    label_mode='categorical'                     
)

#데이터개수와 라벨링 개수 자동 인식 
test_dataset = image_dataset_from_directory( 
    new_base_dir/"test",
    image_size=(180, 180), 
    batch_size=16,
    label_mode='categorical'                     
)

#이미 학습된 컨브넷을 가져온다 
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

conv_base = VGG19(
    weights="imagenet",
    include_top=False, 
    input_shape=(180, 180, 3)
)

conv_base.summary()

#합성곱 신경망이 하는일이 특성추출 - 중요한 특성만 간단하게 
#우리데이터를 넣어서 그 데이터 특성만 추출해서 학습에 적용하자 
import numpy as np
import matplotlib.pyplot as plt 
i=1
def get_features_and_labels(dataset):
    all_features=[]
    all_labels=[]
    for images, labels in dataset: #한번에 32건씩
        preprocessed_images = preprocess_input(images)
        # plt.imshow(images[0])
        # plt.show()
        # plt.imshow(preprocessed_images[0])
        # plt.show()
        features = conv_base.predict(preprocessed_images)#특성추출
        all_features.append( features )
        all_labels.append( labels)
        # if i==1:
        #     break 
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
print( train_features.shape )
print( train_labels.shape)

#CNN 층 끝났고 
import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow import keras 

data_argumentation = keras.Sequential([
    layers.RandomFlip("horizontal") ,
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.2) 
])

inputs = tf.keras.Input(shape=(5,5,512))
#x = data_argumentation(inputs)
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x) #데이터에 일부러 noise를 준다. 과대적합을 막기 위해서 
#기존 파라미터들을 1/2를 없애버린다. 
outputs = layers.Dense(6, activation="softmax")(x)

network = keras.Model(inputs, outputs)
network.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
history = network.fit( 
    train_features, 
    train_labels,
    epochs=5)
