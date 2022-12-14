import tensorflow as tf
import os
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.constant(10)
b = tf.constant(20)     # 파이썬 형을 = > tensor로 바꿔준다.

print(a)
print(b)

# rank - 차원의 수(variable) a:Any
print(tf.rank(a))

print(tf.cast(a, tf.float32)) # 형 전환하기
a = tf.cast(a, tf.float32) # 형을 실수로 전환 받기
b = tf.cast(b, tf.float32) # 형을 실수로 전환 받기
print(a.dtype)

print(tf.math.add(a, b))
print(tf.math.subtract(a, b))
print(tf.math.multiply(a, b))   # 나머지 구하기
print(tf.math.divide(a, b))     # 몫나누기
# 실수는 나머지에 대해 연산이 불가하다
a1 = tf.cast(a, tf.int32)   # 형을 실수에서 정수로 다시 전환 받기
b1 = tf.cast(b, tf.int32)   # 실수 => 정수
print(tf.math.floormod(a1, b1))   # 나머지 구하기