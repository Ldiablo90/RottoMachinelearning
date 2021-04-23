import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
dataset


for elem in dataset:
  print(elem.numpy())
  

it = iter(dataset)

print(next(it).numpy())

print("dataset")

print(dataset.reduce(0, lambda state, value: state + value).numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

print("dataset1")

print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]),tf.random.uniform([4,100], maxval=100, dtype=tf.int32)))
print("dataset2")
print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print("dataset3")
print(dataset3.element_spec)

dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0,0],[1,2]], values=[1,2], dense_shape=[3,4]))
print("dataset4")
print(dataset4.element_spec)
print(dataset4.element_spec.value_type)

