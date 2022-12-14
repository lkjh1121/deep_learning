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

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
conv_base = ResNet50( 
    weights="imagenet",
    include_top=False, #top - 완전연결망은 배제하고  
    input_shape=(180, 180, 3) #이미지크기 - 180은 내마음, 데이터셋과 크기가 맞아야 한다 
)

conv_base.summary() 

# """
# 1.특성추출
#   1-1.  데이터 증식을 하지 않는 경우 : 단순히 특성만 추출한다  
#   1-2.  데이터 증식을 하는 경우 

# 2.미세조정 

# """


from keras.utils import image_dataset_from_directory
import pathlib
#from keras.applications.vgg19 import preprocess_input
 

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
def get_features_and_labels(dataset): #train_ds, test_ds, valid_ds
    #dataset만들때 batch_size : 16개임
    all_features=[] #모든 특성을 저장함 
    all_labels=[] 
    i=1
    for images, labels in dataset:
        # print(images )
        # print(labels)
        preprocessed_images =   preprocess_input(images)
        features = conv_base.predict(preprocessed_images, verbose=0)
        all_features.append( features)
        all_labels.append(labels)

        # plt.imshow(images[1])#원래이미지는 출력 안됨
        # plt.show()
        # plt.imshow(preprocessed_images[1])
        # plt.show()
        # if i==1:
        #     break
    
    return np.concatenate(all_features ), np.concatenate(all_labels)

#1.데이터셋 먼저 만들고
train_ds, test_ds, valid_ds = createDataset()
#2.특성을 추출하고 
train_features, train_labels = get_features_and_labels(train_ds)
test_features, test_labels = get_features_and_labels(test_ds)
valid_features, valid_labels = get_features_and_labels(valid_ds)
#3.모델을 만들어서 학습하자 
#conv_base.summary() 맨상위층이  6, 6, 2048
inputs = keras.Input(shape=( 6, 6, 2048))

from keras import layers

x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x) #절반의 데이터를 버린다. 과대적합을 막으려고 
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model( inputs, outputs) #모델만들기 

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit( 
    train_features, train_labels, 
    epochs=5, 
    validation_data = ( valid_features, valid_labels)
)










