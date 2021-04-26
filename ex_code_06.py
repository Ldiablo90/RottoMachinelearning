import tensorflow as tf

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file,'rb').read().decode(encoding='utf-8')
print('txt is len : {} ex'.format(len(text)))

print(text[:250])

vocab = sorted(set(text))
print('static word : {} ex'.format(len(vocab)))

char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s} : {:3d},'.format(repr(char),char2idx[char]))
print('......\n')


print ('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

seq_length = 100
examples_per_epoch = len(text)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
