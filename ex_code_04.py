import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))

print(train_data[0])

print(len(train_data[0]), len(train_data[1]))
