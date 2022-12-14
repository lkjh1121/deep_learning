import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# 고정 메모리
# 텐서플로는 인덱스나 슬라이싱을 이용해서 데이터를 수정할 수 없다.
m1 = tf.constant(np.arange(1, 6))
# m1[0] = 10 # 갑을 넣을수가 없다, 절대 불가능하다.
# m1[0:4] = 10 # numpy는 허용가능하다 

m2 = tf.Variable(np.arange(1, 6))
print(m2)
# m2[0]=10 # 변수지만 안된다.

# 변수
m2.assign([4,5,6,7,8])
print(m2)
