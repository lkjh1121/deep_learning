from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers

import tensorflow as tf
# 즉시 실행 모두 종료 - 메모리 부족 시 
tf.compat.v1.disable_eager_execution()
# 본 데이터셋은 25000개의 train data, 25000개의 test data 로 긍정, 부정인 리뷰의 수가 동일하게 균형이 맞춰져 있다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[:5])
print(train_labels[:5])

# 가공작업을 해서 보여준다. get_word_index()
# 영화 평 문장 -> token화 시킨다( token화 - 단어 단위로 쪼갠다 )
# 읽는 순서대로 movie:1, like:2, hate:3, actor:4, action:5
# 문장을 위에 정의된 index로 바꿔치기 - 시퀀스(문자열을 취급할 때 시퀀스로 바꿔야 한다)
word_index = imdb.get_word_index()


import numpy as np
# 시퀀스 => 벡터화 한다, 10000개만 쓰고 나머지 버림
def vectorize_sequence(sequenceList, dimension=10000):
    results = np.zeros((len(sequenceList), dimension)) # 0으로 채워진 배열 생성
    # 0으로 채워진 행은 문장의 개수, 열은 10000개까지 배열을 생성한다(2차원배열)
    # enumerate(리스트 타입, 반복적인 객체) - 인덱스와 요소를 반환한다
    for i, sequence in enumerate(sequenceList):
        results[i, sequence] = 1. # i -문장번호, sequence는 문장을 구성하는 index들

    return results

X_train = vectorize_sequence(train_data)
X_test = vectorize_sequence(test_data)

print(X_train[:3,:20])

# 시퀀스를 벡터라이징 할 때는 못 쓴다
# from tensorflow.keras.utils import to_categorical
# X_train2 = to_categorical(train_data[0])

print(type(train_labels))
print(train_labels[:20])

y_train = train_labels
y_test = test_labels

# train 데이터 셋을 train과 validation 셋으로 나눈다.
# 총 25000개 데이터셋 중
X_val = X_train[:10000] # 검증 셋이 10000
partial_x_train = X_train[10000:] # 나머지 훈련셋
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid')) # 2진분류

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train,partial_y_train, epochs=5, batch_size=100,validation_data=(X_val,y_val))

res1 = model.evaluate(X_train,y_train)
res2 = model.evaluate(X_test,y_test)
print(res1)
print(res2)

# 예측하기
pred = model.predict(X_test)
print(pred)
print(pred.shape)

def changeData(pred):
    for i in range(len(pred)):
        if pred[i] < 0.8:
            pred[i] = 0
        else:
            pred[i]=1
    return pred
pred2 = changeData(pred)
for i in range(0,40):
    print(pred[i], y_test[i])

