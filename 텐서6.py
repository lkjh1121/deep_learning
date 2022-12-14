import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
# # 고차원 상수 만들기

# 문제1. 1~100까지 원소를 갖는 1차원 텐서 만들기
a = tf.constant(np.arange(1,101))
print(a)

# 답.1
t1 = tf.constant(np.arange(1,101))
print(t1)

# 문제2. 1~100까지중에서 짝수의 원소를 갖는 1차원 텐서 만들기
b = tf.constant([np.arange(2,101, 2)])
print(b)

# 답.2
t2 = tf.constant(np.arange(2,101, 2))
print(t2)

# 문제3. 1~100까지중에서 홀수의 원소를 갖는 1차원 텐서 만들기
c = tf.constant([np.arange(1,100, 2)])
print(c)
print("--------------------------------------------------")

# 답.3
t3 = tf.constant(np.arange(1,101, 2))
print(t3)

# 문제4. 1~100까지 원소를 갖는 2차원 텐서 만들기, 4 by 25

# 답.4
t4 = tf.constant(np.arange(1,101).reshape(4, 25))
print(t4)

t4 = tf.constant(np.arange(1,101).reshape(25, 4))
print(t4)

### 인덱싱

t1 = tf.constant(np.arange(10, 101, 10))
print(t1)

print(t1[:5])
print(t1[::])
print(t1[5:])
print(t1[3:7:])
print(t1[::2])
print(t1[::-1]) # 역순으로 출력하기
print(t1[0], t1[1], t1[9]) # 0~10까지 10은 없음 9 까지 출력

m1 = tf.reshape(t1, [2, 5])
print(m1)
print(m1[0]) # m1[0, :]
print(m1[1]) # m1[1, :]
print(m1[0, 0])
print(m1[0, 1:4])

m2 = tf.constant(np.arange(1,25).reshape(4,6))
print(m2)
print(m2[0:2, 1:3])
print(m2[0:2, :])
print(m2[:, 2:4])

 # 6 by 4 = 24개
print(tf.reshape(m2, [2, 12]))
print(tf.reshape(m2, [2, -1])) # -1을 쓰면 자동으로 하겠다.

print(tf.reshape(m2, [4, -1]))

# 3차원 전환
print(tf.reshape(m2, [3, 2, -1]))

# 4차원 전환
print(tf.reshape(m2, [3, 2, 2, -1]))

# 차원 추가하기
print(tf.constant(m2, [-1, 1]))

