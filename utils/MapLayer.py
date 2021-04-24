from keras import backend as K
from keras.layers import Layer
import keras
from tensorflow.sparse import SparseTensor
import tensorflow as tf

# here I construct my initializer first


class MyMapLayer(Layer):

    def __init__(self, output_dim, mapping, **kwargs):
        self.output_dim = output_dim
        #self.mapping = mapping
        def my_init(shape, dtype=None):
            return mapping
        self.my_init = my_init
        super(MyMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=keras.initializers.glorot_normal(seed=None),
                                      trainable=True)
        self.kernel2 = self.add_weight(name='kernel2', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.my_init,
                                      trainable=False)
        super(MyMapLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        W = self.kernel * self.kernel2
        
        return K.dot(x, W)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)