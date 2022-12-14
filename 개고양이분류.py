import numpy as np 
import os 
import PIL.Image as pilimg  
import pandas as pd 
import imghdr # 파일의 이미지 종류 확인
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(1234)
# 시드고정 
SEED=1
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# 폴더에서 이미지 파일을 읽어서 numpy 배열로 만들어서 저장하자
# PIL.Image - 이미지를 읽어서 쉽게 numpy 배열로 바꾼다.
# imghdr - 이미지 종류 확인( 파일의 속성 ), 파일 정보에서 파일의 성격을 확인 # 각 구역별 공간의 대한 자기 파일정보(속성)
def makeData(keyworld="cat", label=0):
    # 파일이 있는 위치
    path = r".\datasets\cats_and_dogs\train"
    fileList = os.listdir(path) # 해당 경로에 있는 파일과 폴더등 모든 목록을 읽어온다.
    i=1
    dataList = [] # 이미지 저장할 list
    labelList = [] # 라벨 저장할 리스트

    for filename in fileList:
        if keyworld not in filename:
            continue # 이 아래 문장을 수행하지 않고 for문으로 다시 되돌아간다.

        print(filename)
        if i==50:
            print(f"{i}번째 처리중")
        i = i + 1
        try:
            # 파일의 종류를 확인한다.
            kind = imghdr.what(path + "/" + filename)
            # print(kind)
            if kind in ["gif", "png", "jpg", "jpeg"]: # 이미지 파일이면
                img = pilimg.open(path + "/" + filename)
                resize_img = img.resize((150, 150)) # 사진의 크기를 줄인다.
                pixel = np.array(resize_img) # 자른 이미지를 numpy 배열로 변환한다.
                # 가끔 3차원 아닌 이미지가 따라들어올때가 있는데 
                if pixel.shape == (150, 150, 3): # 3은 색상 값
                    # 데이터를 list에 저장하고, 라벨링 작업도
                    dataList.append(pixel)
                    labelList.append(label)
        except Exception as e: # 에러메세지
            print(e)
            # os.remove(path+"/"+filename)

    # numpy 배열을 저장하자.
    np.savez(f"{keyworld}.npz", data = dataList, target = labelList)

    # 확장자는 바꾸지 못한다. 키1=값1, 키2=값2 dict 타입형태로 저장한다.
def dataLoad():
    cat = np.load("cat.npz")
    dog = np.load("dog.npz")

    data1 = cat["data"] # key 값을 이용해서 dataList를 읽어온다.
    target1 = cat["target"]

    data2 = dog["data"] # key 값을 이용해서 dataList를 읽어온다.
    target2 = dog["target"]

    # 데이터를 행으로 합친다. 튜플(data1, data2, data3....), 축(axis)
    data = np.concatenate((data1, data2), axis=0)
    target = np.concatenate((target1, target2), axis=0)
    return data, target

    print(data[:3])
    print(target[:30])
    
def study():
    data, target = dataLoad()
    X_train, X_test, y_train, y_test = train_test_split(data, target,
    random_state=1, test_size=0.3)
    print(y_train[:30])
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # 1. 4차원 => 2차원
    X_train = X_train.reshape((X_train.shape[0], 150*150*3,))
    X_test = X_test.reshape((X_test.shape[0], 150*150*3,))
    # 정규화
    X_train = X_train/255
    X_test = X_test/255
    # 3. 출력데이터 (원핫 인코딩) 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 모델 만들기
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(150*150*3,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")])

    # 컴파일 과정을 거친다. 최적화 함수 : sgd, adam, rmsprop...
    network.compile(optimizer='sgd', # optimizer 최적화 함수
                    loss='categorical_crossentropy', # 손실함수 - 다중분류일때 고정으로 사용한다
                    metrics=['accuracy'])            # 평가항목, 정확도

    # 학습
    hist = network.fit( X_train, y_train, 
                        epochs=5, # 학습횟수
                        batch_size=100) # 한번에 메모리 불러오는 데이터의 양

    train_loss, train_acc = network.evaluate(X_train, y_train)
    test_loss, test_acc = network.evaluate(X_test, y_test)

    print(f"훈련셋 손실 : {train_loss} 정확도:{train_acc}")
    print(f"테스트셋 손실 : {test_loss} 정확도:{test_acc}")


if __name__ == "__main__":
    while True:
        print("1.데이터만들기")
        print("2.데이터불러오기")
        print("3.학습하기")
        print("0.종료")

        sel = input("선택 : ")
        if sel=="1":
            makeData("cat", 0)
            makeData("dog", 1)
        elif sel=="2":
            data, target = dataLoad()
        elif sel=="3":
            study()    
        elif sel=="0":
            break  #while문 종료