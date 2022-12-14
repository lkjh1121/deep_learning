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
    batch_size=32 ,
    label_mode='categorical'    ########## 추가, default-int           
)

validation_dataset = image_dataset_from_directory( 
    new_base_dir/"train",
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(180, 180), 
    batch_size=32,
    label_mode='categorical'                     
)


#데이터개수와 라벨링 개수 자동 인식 
test_dataset = image_dataset_from_directory( 
    new_base_dir/"test",
    image_size=(180, 180), 
    batch_size=32,
    label_mode='categorical'                     
)

#Sequential 만들어도 되는데 함수 API를 사용해서 만들기 
from tensorflow import keras 
from tensorflow.keras import layers 

#데이터 증식을 위해서 (이미지를 왜곡시켜셔 강제로 새로운 이미지를 만든다. )
#원래 이미지 그대로 있고, 증식시켜서 가져온다 
data_argumentation = keras.Sequential([
    layers.RandomFlip("horizontal") ,
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.2) 
])

inputs = keras.Input( shape=(180, 180, 3) ) # shape에는 이미지크기를 지정한다 
x  = data_argumentation(inputs)
x = layers.Rescaling(1./255)(x) #정규화 
x = layers.Conv2D( filters=32, kernel_size=3, activation="relu")(x)
#filters - 만들어내야할 필터 개수, kernel_size=차원 
x = layers.MaxPooling2D(pool_size=2)(x) #위에서 나온 내용중 중요한것만 추린다 
x = layers.Conv2D( filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x) 
x = layers.Conv2D( filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x) 
x = layers.Conv2D( filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x) 
x = layers.Conv2D( filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x) #완전연결층이랑 연결하기 
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(6, activation='softmax')(x) #라벨이 2개임, 2진분류일때 
model = keras.Model(inputs=inputs, outputs = outputs )
#라벨이 cat-0이고 dog가 1임 라벨 1이 될 확률을 가져온다 
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#32
history = model.fit(
    train_dataset.repeat(5), 
    epochs=100,
    validation_data = validation_dataset
)

#데이터셋은 한번에 예측이 안된다.
# i=0 
# match_cnt=0
# import numpy as np 
# for images, labels in test_dataset:
#     y_pred = model.predict( images)
#     print( y_pred.shape )
#     for j in range(0, y_pred.shape[0]):
#         #pos = np.argmax(y_pred[j])
#         print(np.argmax(y_pred[j]), np.argmax(labels[j].numpy()))
    
#tf.data.experimental.cardinality(test_dataset).numpy() - batch_size

import tensorflow as tf 
print("전체개수 : ", i,   "일치한개수",match_cnt)

#이미지 증식 아니고 그냥 폴더에서 읽기 

#작은 네트워크(모델)을 만들어서(과소적합)=> 늘려가며 과대적합 상태를 만들자

#데이터셋을 사용할때는 데이터를 폴더를 만들고 폴더에 넣어주면 알아서 라벨링을 한다. 
#train - dog, cat 
#model만들어서 fit 함수에 train_dataset를 던져주면 데이터 알아서 읽어오고, 라벨링도 알아서 한다 


#차트그리기
import matplotlib.pyplot as plt 

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(accuracy)+1)

#정확도
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure() 

#손실도
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation loss")
plt.legend()
plt.show() 





