import time, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

from utils import *

# Much of this follows https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=2grYvXLzJYkP

def get_stategy_TPU():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)

def get_distrib_dataset(batch_size=32):
    steps_per_train = 60000 // batch_size
    steps_per_test = 10000 // batch_size

    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

    train_ds = strategy.experimental_distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, is_training=True))

    test_ds = strategy.experimental_distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, is_training=False))

    return steps_per_train, steps_per_test, train_ds, test_ds

def get_dataset(batch_size, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                            as_supervised=True, try_gcs=True)

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    dataset = dataset.map(scale)
    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    return dataset

def create_model():
    return tf.keras.Sequential(
        [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)])


@timerfunc
def train(model, train_loss, train_acc, optimizer, loss_object):
    @tf.function
    def train_step(iterator):
        def train_step_fn(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_object(labels, logits, from_logits=True)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

            train_loss.update_state(loss * strategy.num_replicas_in_sync)
            train_acc.update_state(labels, logits)

        strategy.run(train_step_fn, args=(next(iterator),))

    train_iterator = iter(train_ds)
    train_loss.reset_states()
    train_acc.reset_states()

    for step in range(steps_per_train):
        train_step(train_iterator)

    return float(train_loss.result()), 100*float(train_acc.result())

@timerfunc
def test(model, test_loss, test_acc, loss_object):
    @tf.function
    def test_step(iterator):
        def test_step_fn(inputs):
            images, labels = inputs
            logits = model(images, training=False)
            loss = loss_object(labels, logits, from_logits=True)
            loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)

            test_loss.update_state(loss * strategy.num_replicas_in_sync)
            test_acc.update_state(labels, logits)

        strategy.run(test_step_fn, args=(next(iterator),))

    test_iterator = iter(test_ds)
    test_loss.reset_states()
    test_acc.reset_states()

    for step in range(steps_per_test):
        test_step(test_iterator)

    return float(test_loss.result()), 100*float(test_acc.result())


def run_each_model(model_id, config, tqdm_args=dict(), print_perf=True):

    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    with strategy.scope():
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_object = tf.keras.losses.sparse_categorical_crossentropy
        train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        test_acc = tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)

    bm_dict = init_benchmark_stats(model_id, num_epoch)

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
    parser = common_parser('Tensorflow Benchmark TPU')

    args = parser.parse_args()

    save_path = args.save_path
    print_perf = args.print_perf
    overwrite_file = args.overwrite
    args = vars(args)

    config = dict(
        library = 'TensorFlow',
        device= 'TPU',
        **args
    )

    del config['save_path']
    del config['print_perf']
    del config['overwrite']

    main_tqdm_args = dict(
        position=0, leave=False,
        desc="Main", ncols=50,
        colour='green'
    )

    each_tqdm_args = dict(
        position=1, leave=False, ncols=50
    )

    out_file = config_filename(config, save_path)

    batch_size = config['batch_size']
    num_run = config['num_run']
    strategy = get_stategy_TPU()
    steps_per_train, steps_per_test, train_ds, test_ds = get_distrib_dataset(batch_size)

    dfs = []

    for i in tqdm(range(num_run), **main_tqdm_args):
        model_id = '%02d' %(i)
        dfs.append(run_each_model(
            model_id, config,
            each_tqdm_args, print_perf
        ))

    df = pd.concat(dfs, ignore_index=True).assign(**config)

    write_csv(df, out_file, overwrite_file)
