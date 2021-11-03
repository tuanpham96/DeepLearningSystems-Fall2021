
import math, os, time, pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf

from src.dataset import *
from src.model import *

def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss_ = loss_object(target, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = tf.cast(step, tf.float32) * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def configure_datafiles(data_path, train_filename, nonbreaking_filenames):
    return dict(
        train   = os.path.join(data_path, train_filename),
        input   = os.path.join(data_path, nonbreaking_filenames[0]),
        target  = os.path.join(data_path, nonbreaking_filenames[1]),
    )


def compile_model(transformer, model_config):
    # Define metric
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    # Create LR scheduler and Adam optimizer
    leaning_rate = CustomSchedule(model_config['d_model'])
    optimizer = tf.keras.optimizers.Adam(leaning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[train_accuracy]
    )

def fit_model_with_callbacks(transformer, dataset, model_name, num_epochs=1, profile_batch='500,520'):
    logs = "logs/" + model_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = logs,
        histogram_freq = 1,
        profile_batch = profile_batch
    )

    transformer.fit(
        dataset,
        epochs=num_epochs,
        callbacks = [tboard_callback]
    )