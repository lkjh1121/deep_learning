# pip install gdown 
# import gdown
# gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')

from tensorflow.keras.utils import image_dataset_from_directory 
#디렉토리로 부터 데이터를 불러서 사용한다 

"""
train_dataset = image_dataset_from_directory( 
    "디렉토리명",           -- 파일을 불러올 경로명
    image_size=(180, 180), -- 이미지 크기(이미지 불러오면서 자동으로 크기를 맞춘다)
    batch_size=32          -- 한번에 불러올 이미지 개수, 컴퓨터 메모리가 작으면 이 수치를 
                              낮추자   
)
"""
import os, shutil, pathlib 

#train폴더에 있는 dog와 cat을 각각 폴더만들고 파일을 찾아서 옮기도록 하자 
base_dir = pathlib.Path("./dogs-vs-cats/train")
#이미지 일부만 옮긴다 
new_base_dir = pathlib.Path("./dogs-vs-cats/cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        #/dogs-vs-cats/cats_vs_dogs_small/train/cat 
        dir = new_base_dir/subset_name/category
        os.makedirs(dir) #경로가 없으면 만든다. 
        #파일 이동하기 
        fnames = [f"{category}.{i}.jpg" for  i in range(start_index, end_index)]
        #파일이름이 cat.1.jpg, cat.2.jpg형태임 - 파일목록을 만들어냄 
        for fname in fnames:
            shutil.copyfile( src=base_dir/fname, dst=dir/fname)

#make_subset("train", start_index=0, end_index=1000)
#make_subset("validation", start_index=1000, end_index=1500)
#make_subset("test", start_index=1500, end_index=2000)

train_dataset = image_dataset_from_directory( 
    new_base_dir/"train",
    image_size=(180, 180), 
    batch_size=32              
)
#데이터개수와 라벨링 개수 자동 인식 
test_dataset = image_dataset_from_directory( 
    new_base_dir/"test",
    image_size=(180, 180), 
    batch_size=32              
)

validation_dataset = image_dataset_from_directory( 
    new_base_dir/"validation",
    image_size=(180, 180), 
    batch_size=32              
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
outputs = layers.Dense(1, activation='sigmoid')(x) #라벨이 2개임, 2진분류일때 
model = keras.Model(inputs=inputs, outputs = outputs )
#라벨이 cat-0이고 dog가 1임 라벨 1이 될 확률을 가져온다 
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#32
history = model.fit(
    train_dataset.repeat(5), 
    epochs=10,
    validation_data=validation_dataset
)

#데이터셋은 한번에 예측이 안된다.
i=0 
match_cnt=0
for images, labels in test_dataset:
    y_pred = model.predict( images)
    for j, pred in enumerate(y_pred):
        if pred>=0.5:
            pred=1
        else:
            pred=0
        if pred==labels[j].numpy():
            match_cnt+=1
        #print(pred, labels[j].numpy())
        i=i+1

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

