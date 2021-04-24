"""
This is a global feature selection layer that adds weights for each feature.
"""
from keras import backend as K
from keras.layers import Layer
from keras.regularizers import l1
from keras.activations import relu
import keras
from tensorflow.sparse import SparseTensor
import tensorflow as tf
from keras.constraints import NonNeg



class FSLayer(Layer):
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(FSLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # print("This is input_shape", input_shape) -> (None, 6061)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],),
                                      #initializer=keras.initializers.glorot_normal(seed=None),
                                      #initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                                      #initializer=keras.initializers.Ones(),
                                      initializer=keras.initializers.he_normal(seed=None),
                                      regularizer=l1(0.001), # 0.001
                                      trainable=True,
                                      constraint=NonNeg())

        super(FSLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape