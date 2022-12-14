import tensorflow as tf
import os
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 쓸데없는 경고 안나오게, window 11 버전은 경고 안나옴
# window 10버전만 경고 창이 뜬다.
print(tf.config.list_physical_devices('GPU'))
