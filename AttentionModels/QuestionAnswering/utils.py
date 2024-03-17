import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_length=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=self.max_length, depth=self.d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# Dummy enconder block. Will be replaced by learner implementation
class EncoderBlock(tf.keras.Model):
    def __init__(self, d_model, d_ff, n_heads, dropout, ff_activation='relu'):
        super().__init__(self)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.ff_activation = ff_activation
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        return x