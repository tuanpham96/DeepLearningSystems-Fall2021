import math
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Sequential

@tf.function
def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)
    scaled_product = product / tf.math.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32))
    scaled_product += (mask * -1e9)
    return tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)

class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)
        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size): # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size, -1, self.n_heads, self.d_head) # Set the dimension of the projections
        splited_inputs = tf.reshape(inputs, shape=shape) # (batch_size, seq_length, nb_proj, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) # (batch_size, nb_proj, seq_length, d_proj)

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]
        queries = self.split_proj(self.query_lin(queries), batch_size)
        keys = self.split_proj(self.key_lin(keys), batch_size)
        values = self.split_proj(self.value_lin(values), batch_size)
        attention = scaled_dot_product_attention(queries, keys, values, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention,shape=(batch_size, -1, self.d_model))
        return self.final_lin(concat_attention)

class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model): # pos: (seq_length, 1) i: (1, d_model)
        angles = 1 / np.power(10000., (2.0*(i//2)) / (d_model * 1.0))
        return pos * angles # (seq_length, d_model)

    def call(self, inputs): # input shape batch_size, seq_length, d_model
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...] * 1.0
        return inputs + pos_encoding

class EncoderLayer(layers.Layer):
    def __init__(self, FFN_units, n_heads, dropout_rate, act_fun):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.act_fun = act_fun

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.multi_head_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffns_and_dropout = Sequential([
            layers.Dense(units=self.FFN_units, activation=self.act_fun),
            layers.Dense(units=self.d_model),
            layers.Dropout(rate=self.dropout_rate)
        ])
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs,inputs,inputs,mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)
        outputs = self.ffns_and_dropout(attention, training=training)
        outputs = self.norm_2(outputs + attention)

        return outputs

class Encoder(layers.Layer):

    def __init__(self,n_layers,FFN_units,n_heads,dropout_rate,vocab_size,d_model,act_fun,name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [EncoderLayer(FFN_units, n_heads, dropout_rate,act_fun) for _ in range(n_layers)]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs) * tf.math.sqrt(self.d_model * 1.0)
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs, mask, training)
        return outputs

class DecoderLayer(layers.Layer):

    def __init__(self, FFN_units, n_heads, dropout_rate, act_fun):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.act_fun = act_fun

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.multi_head_causal_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.multi_head_enc_dec_attention = MultiHeadAttention(self.n_heads)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffns_and_dropout = Sequential([
            layers.Dense(units=self.FFN_units, activation=self.act_fun),
            layers.Dense(units=self.d_model),
            layers.Dropout(rate=self.dropout_rate)
        ])
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_causal_attention(inputs,inputs,inputs,mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention + inputs)
        attention_2 = self.multi_head_enc_dec_attention(attention, enc_outputs, enc_outputs, mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)
        outputs = self.ffns_and_dropout(attention_2, training=training)
        outputs = self.norm_3(outputs + attention_2)

        return outputs

class Decoder(layers.Layer):

    def __init__(self,n_layers,FFN_units,n_heads,dropout_rate,vocab_size,d_model,act_fun,name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.dec_layers = [DecoderLayer(FFN_units,n_heads,dropout_rate,act_fun) for _ in range(n_layers)]

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs) * tf.math.sqrt(self.d_model * 1.0)
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        for i in range(self.n_layers):
            outputs = self.dec_layers[i](outputs,enc_outputs,mask_1,mask_2,training)
        return outputs

class Transformer(tf.keras.Model):

    def __init__(self,vocab_size_enc,vocab_size_dec,d_model,n_layers,FFN_units,n_heads,dropout_rate,act_fun = 'relu',name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.encoder = Encoder(n_layers,FFN_units,n_heads,dropout_rate,vocab_size_enc,d_model,act_fun)
        self.decoder = Decoder(n_layers,FFN_units,n_heads,dropout_rate,vocab_size_dec,d_model,act_fun)
        self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_ouput")

    def create_padding_mask(self, seq): #seq: (batch_size, seq_length)
        mask = tf.where(tf.math.equal(seq, 0), 1.0, 0.0)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, inputs, training):
        enc_inputs, dec_inputs = inputs
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_mask_1, dec_mask_2, training)
        outputs = self.last_linear(dec_outputs)
        return outputs

    def train_step(self, data):
        enc_inputs, targets = data
        dec_inputs = targets[:, :-1]  # Set the decoder inputs
        dec_outputs_real = targets[:, 1:]  # Set the target outputs, right shifted

        with tf.GradientTape() as tape:
            predictions = self((enc_inputs, dec_inputs), True) # Call the transformer and get the predicted output
            loss = self.compiled_loss(dec_outputs_real, predictions, regularization_losses=self.losses) # Calculate the loss

        # Update the weights and optimizer
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics
        self.compiled_metrics.update_state(dec_outputs_real, predictions)
        return {m.name: m.result() for m in self.metrics}
