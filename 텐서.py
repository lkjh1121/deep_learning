import tensorflow as tf
import os
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 쓸데없는 경고 안나오게, window 11 버전은 경고 안나옴
# window 10버전만 경고 창이 뜬다.

a = 1
b = 2
c = tf.math.add( a, b ) # 텐서 객체를 생성한다.
print(c)
print(c.numpy())        # tensor = > numpy로 변경한다.


"""
y = ax + b  : 선형회귀

a와 b를 찾아내자

실제 자연계에서 얻어지는 수치가 딱 맞아 떨어지는 경우는 없다
y = 2x + 1          a =  2 ,    b = 1 
x = [1,2,3,4,5,6]
y = [3,5,7,9,11,13]

ex) 부동산 가격 예측 수식을 찾아내고 싶다
y = ax + b      시작할때 a, b 도 모른다.
랜덤하게 a 하고 b를 놓고 연산을 수행
바꿔가면서 a 하고 b를 놓고 연산을 수행
바꿔가면서 a 하고 b를 놓고 연산을 수행
바꿔가면서 a 하고 b를 놓고 연산을 수행
바꿔가면서 a 하고 b를 놓고 연산을 수행

최소제곱 오차법 - 손실
오차 = (실제 값 - 예측 값)
+, - 나오고 손실을 오차들의 합을 구하면 손실이 0이 된다.
1. 절대값구해서 평균 구하기 (mae) mean absoute error
2. 제곱을 구해서 평균 구하기 (mse) mean squared error
   (수학자들이 좋아하는 공식)

제곱이기 떄문에 a (weight)를 모두 모으면 대접모양이 나온다.
대접모양의 기울기를 그때 그때 찾을때(자동미분) 가장 오차가 적은 지점을 찾아간다. # 경사하강법


"""