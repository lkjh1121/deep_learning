import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# Y=3*X - 1
g = tf.random.Generator.from_seed(2020)
X = g.normal(shape=(10,))
Y = 3*X -2
print("X : ", X.numpy())
print("Y : ", Y.numpy())

# 손실 함수
def cal_mas(X, Y, a, b):
    y_pred = a*X + b
    print(f"a : {a} b : {b} y_pred:{y_pred} ")
    # y 데이터 10 벡터
    # y_pred 는 랜덤하게 a(가중치), b(절편)이 주어졌을때 예측 값
    # Y 는 실제값
    error = (Y - y_pred)**2
    print(error)
    return tf.reduce_mean(error)

print(" 손실 값 : ",cal_mas(X, Y , 2, 1))
print(" 손실 값 : ",cal_mas(X, Y , 0, 0))

# 시작 값은 어떻게 넣어놓아도 자동으로 찾아간다.
a = tf.Variable(5.0) # 처음에 주는 값은 랜덤, 기울기(가중치) 
b = tf.Variable(6.0) # 절편

EPOCHS = 200
for epochs in range(1, EPOCHS+1):
    # with 구문은 파일이나 tf.GradientTape 처럼 닫기 작업이 필요한 경우
    # 개발자가 까먹고 안닫은 경우 문제가 된다. 사고 발생나지 않게
    # with 구문에서만 존재하고 with 끝나고 나가면 자동으로 닫힌다.
    with tf.GradientTape() as tape:
        mse = cal_mas(X, Y, a, b)

        grad = tape.gradient(mse, {'a':a, 'b':b})
        d_a, d_b = grad['a'], grad['b']
        
        a.assign_sub(d_a*0.05)
        b.assign_sub(d_b*0.05)

        if epochs %20==0:
            print("EPOCHS %d - MSE %.2f - a:%.2f b:%.2f" %(epochs, mse, a, b))
