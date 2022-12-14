import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# 3차원으로 생성
m1 = [[1,2,3,4],     [5,6,7,8]]
m2 = [[9,10,11,12],  [13,14,15,16]]
m3 = [[17,18,19,20], [21,22,23,24]]

# 3차원 묶음
tensor1 = tf.constant([m1,m2,m3])
print(tensor1)

tensor2 = tf.stack([m1,m2,m3])
print(tensor2)

# 파이썬
v1 = [1,2,3,4]
v2 = [5,6,7,8]
v3 = [9,10,11,12]
v4 = [13,14,15,16]
v5 = [17,18,19,20]
v6 = [21,22,23,24]

# 파이썬으로 3차원 만들기
arr = [[v1, v2],[v3, v4],[v5, v6]] # 3차원 array
tensor3 = tf.constant(arr)
print(tensor3)

print("======================================")
# 파이썬으로 4차원 만들기
tensor4 = tf.stack([tensor1, tensor2])
print(tensor4)

