from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import tensorflow as tf

# 메모리 부족 시 즉시 실행 모두 종료 
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_LOG_LEVEL'] = '3' # 경고레벨 1,2는 무시해도 된다. 화면에 안뜨게 한다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[:5])
print(train_labels[:5])

# 가공작업을 해서 보여준다. get_word_index()
# 영화 평 문장 -> token화 시킨다( token화 - 단어 단위로 쪼갠다 )
# 읽는 순서대로 movie:1, like:2, hate:3, actor:4, action:5
# 문장을 위에 정의된 index로 바꿔치기 - 시퀀스(문자열을 취급할 때 시퀀스로 바꿔야 한다)
word_index = imdb.get_word_index()
# print( type(word_index) )
# print(word_index)

# 내부구조 확인
# def showDictionary(cnt):
#     i=0
#     for key in dict(word_index.items()):
#         if i>=cnt:
#             break
#         i = i+1
#         print(key, dict(word_index.items())[key])
    
# showDictionary(10)

# # 문장을 원상복구 시켜보자  movie:1 dict 타입은 키값으로만 검색이 된다. 값으로는 검색이 안된다.
# # 1         1에 해당하는 단어 1:movie
# temp = [ (value, key) for (key,value) in word_index.items()]
# i=1
# def showReverseDictionary(cnt):
#     i=1
#     for key in dict(temp).keys(): 
#         print(key,dict(temp)[key])
#         if i>=50:
#             break
#         i=i+1

# showReverseDictionary(30)
# reverse_word_index = dict(temp)

# # 문장 하나만 
# wordList = train_data[0] # 첫 번째 문장
# print(wordList)
# for i in range(0, len(wordList)):
#     print(reverse_word_index[wordList[i]])

# sentence = ' '.join( [ reverse_word_index.get(i-3, '?') for i in train_data[0]])
# print(sentence)

# 시퀀스 -> 85000, 3,4: 카테고리, 범주형 자료 - onehot 인코딩

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
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid')) # 2진분류

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train,y_train, epochs=10, batch_size=100)

res1 = model.evaluate(X_train,y_train)
res2 = model.evaluate(X_test,y_test)
print(res1)
print(res2)

