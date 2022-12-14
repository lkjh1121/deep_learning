import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# 행렬
# 2차원 constant로 바뀌면 텐서
a = [[1,2] , [3,4]]
m1 = tf.constant(a) # constant를 넣으면 2차원으로 바뀜
print(m1)

# 단위행렬 - 다 이고 대각선이 1인 행렬이다.
m2 = tf.constant([ [1,0],
                   [0,1] ]) # 단위행렬

print(tf.math.multiply(m1, m2))
"""
    m1[0,0] * m2[0,0], m1[0,1] * m2[0,1]
    m1[1,0] * m2[1,0], m1[1,1] * m2[1,1]    # 그냥 곱하기
"""
# 실제 행렬의 곱
print(tf.matmul(m1, m2))

v1 = [1,2,3,4]
v2 = [5,6,7,8]

a1 = tf.stack( [v1, v2] ) # 벡터를 연결하여 matrix로 만든다
print(a1)

# 브로드 캐스팅 연산 (실시간으로 일일이 하나씩 계산한다.)
# python 지원하지 않는다. numpy 나 tensor에서는 제공한다.
print(a1 + 10 )
print(a1 * 10 )
print(a1 / 10 )

