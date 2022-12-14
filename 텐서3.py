import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

pay_list = [10.0, 20.0, 30.0]
num_list = np.array([10., 10., 10.])

vec1 = tf.constant(pay_list, tf.float32)
vec2 = tf.constant(num_list, tf.float32)

print(vec1)
print(vec2)

r1  = tf.math.add(vec1, vec2)
print(r1)
print(tf.math.add(vec1, vec2))
print(tf.math.multiply(vec1, vec2))
print(tf.math.divide(vec1, vec2))
print(tf.math.floordiv(vec1, vec2))
# print(tf.math.floormod(vec1, vec2))

# 행렬의 합
print(tf.reduce_sum(vec1))
print(tf.reduce_sum(vec2))

print(tf.math.square(vec1)) # 거듭제곱
print(vec1 ** 2) # 파이썬 문법
# 수치연산은 텐서를 사용하는게 훨씬 낫다. ex)
print(tf.math.square(vec1)) # 텐서 내부 교정
print(vec1 * 0.5) # 파이썬 문법

# 브로드 캐스팅
print(vec1 + 5)
vec1 = tf.constant([np.arange(1,21)])
print(vec1[0]) # 인덱싱 출력
# vec1[0] = 7 # 인덱싱 출력은 가능하나 입력은 불가능하다.

print(vec1[:5])
print(vec1[5:])
print(vec1[::-1])
print(vec1[5:0:-1])
print(vec1[vec1 > 5])
print(vec1[ vec1 < 5])
print(vec1[np.logical_and(vec1%2==0, vec1%3==0)])
print(vec1[tf.logical_and(vec1%2==0, vec1%3==0)])