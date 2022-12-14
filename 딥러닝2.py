from tensorflow.keras.datasets import mnist # 손글씨 데이터 셋
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




# 랜덤 값 시드 고정하기
tf.random.set_seed(1234)
iris = load_iris()
(train_image, train_labels), (test_image, test_labels) \
    = mnist.load_data()



# 케라스는 훈련셋과 테스트 셋을 분리해서 보낸다 . 60000, 10000
print( train_image.shape ) 
print( test_image.shape  )

# 피쳐가 1차원 배열 형태이어야 한다. 이미지가 28 by 28 2차원으로 전달 받음
train_image = train_image.reshape(train_image.shape[0], 28 * 28)
print( train_image.shape )

test_image = test_image.reshape(test_image.shape[0], 28 * 28)
print( test_image.shape )

# normalization( 정규화 )
train_image = train_image/255   # 0~255
test_image = test_image/255

# 모델 또는 네트워크 라고도 부른다.
from tensorflow.keras import models, layers

network = models.Sequential() # 간단하게 모델을 만들고자 할 때 사용한다.

# 입력 레이어를 만든다.
inputLayer = layers.Dense(64, activation='relu', input_shape=(28*28, ))
outputLayer = layers.Dense(10, activation='softmax')

"""

 - 입력레이어 - 딥러닝을 시작하는 첫번째 단계이다. 세 개의 매개변수가 있다.
 - 첫번째 매개변수는 이 층의 연산을 수행하고 나올 결과값의 개수이다.
 - 이 개수는 개발자가 마음대로 지정이 가능하다. 너무 작으면 과소적합이 되고
 - 너무 크면 과대적합이 된다. 이 값을 적정하게(적정하게에 대한 단위는 없다.)
 - 보통 8, 16, 24, 32, 128, 256, 512 등등 메모리 크기에 따라 적당히 선택한다.
 - 두번째 매개변수는 연산을 수행하면 그 값이 1이 넘는 큰 값이 된다. 이 값을
 - 0~1 사이에 머무르게 해주는 함수를 활성화 함수라고 하는데 이 함수는 현재 
 - sigmoid 와 relu 둘중 하나를 쓰고 있다. relu가 성능이 더 좋다고 알려져서
 - 현재 relu만 쓴다.
 - activation 가중치와 입력값과 절대 값의 연산의 결과가 1이 넘는다 이 값을
 - 0~1 사이로 머물게 한다. = 현재 렐루를 많이쓴다.
 - 세번째 매개변수는 특성의 개수 - tuple로 넣어야 한다.
 - outputLayer(출력레이어) - 첫 번째 매개변수가 출력 개수 (라벨의 개수) 0 ~ 9 까지 라벨
 - 활성화 함수 - 출력에서는 회귀 & 이중분류 & 다중분류 에 따라 다르다.
              - 다중분류의 경우 softmax 함수를 사용한다.
 - softmax 함수는 결과가 확률로 나온다.
 - 연산이 많이 거듣되다보면 숫자가 엄청 크게 나온다.
 - 확률로 바꾸어서 출력하는 함수가 softmax 함수
 - 중간에 필요하다 싶으면 layer를 많이 만들어서 끼워 넣을 수 있다.
"""

network.add(inputLayer)
network.add(outputLayer)

network.compile(optimizer = "sgd",               # 경사 하강법
                loss="categorical_crossentropy", # 다중분류의 경우
                metrics=["accuracy"])            # 모델 평가 척도(정확도)

# 출력데이터를 - one hot encoding 으로 변환해야 한다.
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_labels[:5])
print(test_labels[:5] )

# 학습 
hist = network.fit(train_image, train_labels,
                    epochs=5, # 학습숫자
                    batch_size=128, # 전체 데이터를 한번에 메모리로)
                    # 로딩할 수 없으므로 128 단위씩 불러오라는 말임
                    # 이 수치도 본인 시스템에 맞춰서 알아서 지정하기
                    # 반환값은 각 에포크마다 의 수행처리 결과를 저장해서 한번에 보내준다.
)

# 평가하기
train_loss, train_acc = network.evaluate(train_image, train_labels)
test_loss, test_acc = network.evaluate(test_image, test_labels)
print("훈련셋", train_loss, train_acc)
print("테스트셋", test_loss, test_acc)

