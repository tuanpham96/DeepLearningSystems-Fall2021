import time, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from utils import *

def get_device(device_type='GPU'):

def get_data_loaders(batch_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds

def init_stat_objs():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc= tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc= tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    return train_loss,

train_ds, test_ds = get_data_loaders(batch_size)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = TFModel()

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accreset_states()
  test_loss.reset_states()
  test_accreset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
        %(epoch, train_loss.result(), train_accresult() * 100))

for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

print(f'Test Accuracy: {test_accresult() * 100:.2f}')

class TFModel(Model):
  def __init__(self):
    super(TFModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

@tf.function
def train_step(images, labels):
    # with tf.device('/CPU:0'):

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acclabels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_acclabels, predictions)