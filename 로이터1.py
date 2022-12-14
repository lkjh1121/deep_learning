from tensorflow.keras.datasets import reuters
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import tensorflow as tf
# 로이터 신문사 기사 섹션 46개, 기사를 읽어 섹션을 맞춘다.

# 즉시 실행 모두 종료 - 메모리 부족 시 
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_LOG_LEVEL'] = '3' # 경고 레벨 1, 2는 무시해도 됨. 화면에 안 뜨게
# 본 데이터셋은 25000개의 train data, 25000개의 test data 로 긍정, 부정인 리뷰의 수가 동일하게 균형이 맞춰져 있다.

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# 처음에는 네트워크를 타고 오고 두번째는 로컬 서 읽는다. 사용자 .keras 에 있다.

print(train_data[:5])
print(train_labels[:5])
print(type(reuters)) # 라벨의 의미가 뭔지 확인 못함

# 문자열을 -> 시퀀스로 만들기 위해 사용했던 인덱스를 준다
# 가공작업을 해서 보여준다. get_word_index()
# 기사 문장 -> token화 시킨다( token화 - 단어 단위로 쪼갠다 )
# 읽는 순서대로 movie:1, like:2, hate:3, actor:4, action:5
# 문장을 위에 정의된 index로 바꿔치기 - 시퀀스(문자열을 취급할 때 시퀀스로 바꿔야 한다)
word_index = reuters.get_word_index()
print(type(word_index))
print(len(word_index))

# 내부구조 확인
def showDictionary(cnt):
    i=0
    for key in dict(word_index.items()):
        if i>=cnt:
            break
        i = i+1
        print(key, dict(word_index.items())[key])
    
showDictionary(10)

# 문장을 원상복구 시켜보자  
# movie:1 dict 타입은 키값으로만 검색이 된다. 값으로는 검색이 안된다.
# 숫자로 검색할 수 있도록
temp = [ (value, key) for (key,value) in word_index.items()] # 단어로만 검색이 되므로
# 키-값을 바꾸어서 숫자를 입력하면 단어가 검색되도록 바꿈
def showReverseDictionary(cnt):
    i=1
    for key in dict(temp).keys(): 
        print(key,dict(temp)[key])
        if i>=50:
            break
        i=i+1

showReverseDictionary(30)
reverse_word_index = dict(temp)

# 문장 하나만 
wordList = train_data[0] # 첫 번째 문장
print(wordList)
for i in range(0, len(wordList)):
    print(reverse_word_index[wordList[i]])

sentence = ' '.join( [ reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(sentence)

# 시퀀스 -> 85000,3,4 : 카테고리 , 범주형 자료 - one hot 인코딩
# 1 3 5 7 8 15 17
# 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1
# results[1, [1,3,5,8,10]]
# 각 문장당 하나의 10000개 짜리 벡터가 할당 된다.
import numpy as np
# # 시퀀스 => 벡터화 한다, 10000개만 쓰고 나머지 버림
def vectorize_sequence(sequenceList, dimension=10000):
    results = np.zeros((len(sequenceList), dimension)) # 0으로 채워진 배열 생성
    # 0으로 채워진 행은 문장의 개수, 열은 10000개까지 배열을 생성한다(2차원배열)
    # enumerate(리스트 타입, 반복적인 객체) - 인덱스와 요소를 반환한다
    for i, sequence in enumerate(sequenceList):
        results[i, sequence] = 1. # i -문장번호, sequence는 문장을 구성하는 index들

    return results

X_train = vectorize_sequence(train_data)
X_test = vectorize_sequence(test_data)

print(X_train[:3,:40])

# 시퀀스를 벡터라이징 할 때는 못 쓴다
# from tensorflow.keras.utils import to_categorical
# X_train2 = to_categorical(train_data[0])

# print(type(train_labels))
# print(train_labels[:20])

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# train 데이터 셋을 train과 validation 셋으로 나눈다.
# 총 25000개 데이터셋 중
X_val = X_train[:1000] # 검증 셋이 10000
partial_x_train = X_train[1000:] # 나머지 훈련셋
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46,activation='softmax')) # 다중분류, softmax
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train,partial_y_train, epochs=3, batch_size=100, validation_data=(X_val,y_val))

res1 = model.evaluate(X_train,y_train)
res2 = model.evaluate(X_test,y_test)
print(res1)
print(res2)

# 예측하기
pred = model.predict(X_test)
print(pred)
print(pred.shape)

def changeData(pred):
    resultList = []
    for i in range(len(pred)):

        resultList.append(np.argmax(pred[i])) # 가장 확률 높은 것의 인덱스
    return resultList
    
# pred2 = changeData(pred)
# print(pred2)
for i in range(0, 40):
    print("예측 : ", np.argmax(pred[i]), "실제값 : ", np.argmax(y_test[i]))


#훈련과 검증  정확도 그리기 
import matplotlib.pyplot as plt 
history_dict = history.history 
acc = history_dict['accuracy']  #훈련셋 정확도
val_acc = history_dict['val_accuracy'] #검증셋 정확도 

print(acc)
print(val_acc)

length = range(1,  len(history_dict['accuracy'])+1 ) #x축만들기 
plt.plot( length, acc , 'ro',  label='Training acc')
plt.plot( length, val_acc , 'b', label='Validation acc')
plt.title("Training and Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()  