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

def get_data_loaders(batch_size=32):
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
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    return train_loss, train_acc, test_loss, test_acc

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

@timerfunc
def train(model, train_loss, train_acc, optimizer, loss_object):
    train_loss.reset_states()
    train_acc.reset_states()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(labels, predictions)

    for images, labels in train_ds:
        train_step(images, labels)

    return float(train_loss.result()), 100*float(train_acc.result())

@timerfunc
def test(model, test_loss, test_acc, loss_object):
    test_loss.reset_states()
    test_acc.reset_states()

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_acc(labels, predictions)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    return float(test_loss.result()), 100*float(test_acc.result())

def run_each_model(model_id, config,
                   tqdm_args=dict(), print_perf=True):

    num_epoch = config['num_epoch']
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss, train_acc, test_loss, test_acc = init_stat_objs()
    model = TFModel()

    bm_dict = init_benchmark_stats(model_id, num_epoch)

    with tf.device(device):
        for i in tqdm(range(num_epoch), desc='ID={}'.format(model_id), **tqdm_args):
            bm_dict['epoch'][i] = i
            (bm_dict['train_loss'][i], bm_dict['train_acc'][i]), bm_dict['train_time'][i] \
                = train(model, train_loss, train_acc, optimizer, loss_object)
            (bm_dict['test_loss'][i], bm_dict['test_acc'][i]), bm_dict['test_time'][i] \
                = test(model, test_loss, test_acc, loss_object)
            if print_perf:
                print_epoch_progress(config, bm_dict, epoch=i)

    df = pd.DataFrame(bm_dict)
    return df

if __name__ == '__main__':
    parser = common_parser('Tensorflow Benchmark GPU')
    args = parser.parse_args()

    save_path = args.save_path
    print_perf = args.print_perf
    overwrite_file = args.overwrite
    args = vars(args)

    config = dict(
        library = 'TensorFlow',
        device= 'GPU',
        **args
    )

    del config['save_path']
    del config['print_perf']
    del config['overwrite']

    out_file = config_filename(config, save_path)

    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise Exception('No cuda/GPU resources available for TF')
    device = '/GPU:0'

    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    num_run = config['num_run']
    train_ds, test_ds = get_data_loaders(batch_size)

    dfs = []

    for i in tqdm(range(num_run), **main_tqdm_args):
        model_id = '%02d' %(i)
        dfs.append(run_each_model(
            model_id, config,
            each_tqdm_args, print_perf
        ))

    df = pd.concat(dfs, ignore_index=True).assign(**config)

    write_csv(df, out_file, overwrite=overwrite_file)
