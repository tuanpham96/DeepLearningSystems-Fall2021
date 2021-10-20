
import math, os, time, pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from src.dataset import *
from src.model import *

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def configure_vocab_sizes(model_config, enc_nwords, dec_nwords):
    config = dict(**model_config) # to avoid shallow copy and unwanted modification
    del config['vocab_size_factor']
    config['vocab_size_enc'] = enc_nwords
    config['vocab_size_dec'] = dec_nwords
    return config

def configure_outputfiles(model_id, output_path = 'output', checkpoint_path='model'):
    return dict(
        results     = os.path.join(output_path, model_id) + '.pkl',
        checkpoint  = os.path.join(checkpoint_path, model_id)
    )

def configure_datafiles(data_path, train_filename, nonbreaking_filenames):
    return dict(
        train   = os.path.join(data_path, train_filename),
        input   = os.path.join(data_path, nonbreaking_filenames[0]),
        target  = os.path.join(data_path, nonbreaking_filenames[1]),
    )

def run_each_model(model_info, data_files, output_files, data_config, model_config, train_config, translator_sentences):

    def_max_vocab_size = 2**14
    data_config['max_vocab_size'] = int(model_config['vocab_size_factor'] * def_max_vocab_size)

    # Load dataset
    dataset, token_dset = load_datasets(data_files, **data_config)

    model_config = configure_vocab_sizes(model_config, token_dset['input']['num_words'], token_dset['target']['num_words'])

    tf.keras.backend.clear_session()

    # Create model and loss
    transformer = Transformer(**model_config)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Data for tracking
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    # Optimizer and scheduler
    leaning_rate = CustomSchedule(model_config['d_model'])
    optimizer = tf.keras.optimizers.Adam(leaning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Create the Checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, output_files['checkpoint'], max_to_keep=train_config['ckpt_max2keep'])

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored.")

    # Define loss function
    def loss_function(target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # Define train loop
    def main_train(dataset, transformer, num_epochs, print_every=50):
        losses, accuracies = [], []

        for epoch in range(num_epochs):
            print("Starting epoch {}".format(epoch+1))
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (enc_inputs, targets)) in enumerate(tqdm(dataset)):
                dec_inputs = targets[:, :-1] # Set the decoder inputs
                dec_outputs_real = targets[:, 1:]  # Set the target outputs, right shifted

                with tf.GradientTape() as tape:
                    predictions = transformer(enc_inputs, dec_inputs, True)
                    loss = loss_function(dec_outputs_real, predictions)

                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

                train_loss(loss)
                train_accuracy(dec_outputs_real, predictions)

                if batch % print_every == 0:
                    losses.append(float(train_loss.result()))
                    accuracies.append(float(train_accuracy.result()))

                    print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                        epoch+1, batch, train_loss.result(), train_accuracy.result()), flush=True)

            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} in {}".format(epoch+1, ckpt_save_path))
            print("Time for 1 epoch: {} secs\n".format(time.time() - start))

        return losses, accuracies

    # Train the model
    T0 = time.time()
    losses, accuracies = main_train(dataset, transformer, train_config['num_epochs'], train_config['print_every'])
    T1 = time.time()
    total_time = (T1 - T0)/60

    # Prediction and translation
    max_length = data_config['max_length']

    tokenizer_inputs = token_dset['input']['tokenizer_corpus']
    sos_token_input = token_dset['input']['sos_token']
    eos_token_input = token_dset['input']['eos_token']

    tokenizer_outputs = token_dset['target']['tokenizer_corpus']
    sos_token_output = token_dset['target']['sos_token']
    eos_token_output = token_dset['target']['eos_token']

    def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
        # Tokenize the input sequence using the tokenizer_in
        inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
        enc_input = tf.expand_dims(inp_sentence, axis=0)

        # Set the initial output sentence to sos
        out_sentence = sos_token_output
        # Reshape the output
        output = tf.expand_dims(out_sentence, axis=0)

        # For max target len tokens
        for _ in range(target_max_len):
            # Call the transformer and get the logits
            predictions = transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
            # Extract the logists of the next word
            prediction = predictions[:, -1:, :]
            # The highest probability is taken
            predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
            # Check if it is the eos token
            if predicted_id == eos_token_output:
                return tf.squeeze(output, axis=0)
            # Concat the predicted word to the output sequence
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def translate(sentence):
        # Get the predicted sequence for the input sentence
        output = predict(sentence, tokenizer_inputs, tokenizer_outputs, max_length).numpy()
        # Transform the sequence of tokens to a sentence
        predicted_sentence = tokenizer_outputs.decode(
            [i for i in output if i < sos_token_output]
        )

        return predicted_sentence

    predicted_sentences = [translate(x) for x in translator_sentences]

    translations = dict(
        input = translator_sentences,
        prediction = predicted_sentences
    )

    with open(output_files['results'], 'wb') as f:
        pickle.dump(dict(
            model_info      = model_info,
            data_files      = data_files,
            output_files    = output_files,
            data_config     = data_config,
            model_config    = model_config,
            train_config    = train_config,
            losses          = losses,
            accuracies      = accuracies,
            total_time      = total_time,
            translations    = translations
            ), f, protocol=pickle.HIGHEST_PROTOCOL)
