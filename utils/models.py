import keras
import os
# Create your first MLP in Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Activation
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l1, l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from utils.MapLayer import MyMapLayer
from utils.FSLayer import FSLayer

def get_model(input_dim, layer_params=[], hyper_params={}, dropout = False):
    # e.g. layer_params = 64, 32
    # create model
    # model = Sequential()
    model_input = Input(shape=(input_dim,), name='Input')
    x = model_input
    reg = keras.regularizers.l1(hyper_params['l1'])
    ############# NEW #############
    if hyper_params['fs_layer']:
        print("FS Layer Enabled")
        x = FSLayer(input_shape=(input_dim, ))(x)
    ############# NEW #############
    print(hyper_params['map_layer'])
    if hyper_params['map_layer']:
        print("Map Layer Enabled")
        output_dim = hyper_params['linking_matrix'].shape[1]   
        x = MyMapLayer(output_dim, hyper_params['linking_matrix'], input_shape=(input_dim, ))(x)
        # previously we add no activation
        out_map = x
        x = Activation('relu', name='out1')(x)
        out1 = x
        # print(x.shape)
        # print(output_dim)
        # if hyper_params['fs_layer']:
        #     x = FSLayer(input_shape=(output_dim, ))(x)

    if len(layer_params) == 0:
        # Here is just the logistic regression
        if hyper_params['map_layer']:
            x = Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='sigmoid', kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']), name='out2')(x)
        else:
            x = Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='sigmoid', kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']), name='out2')(model_input)
        model = Model(inputs=model_input, outputs=x)
        model.summary()
        return model

    i = 0
    for param in layer_params:
        if param <= 0:
            continue
        if (i == 0):
            if hyper_params['map_layer']:
                x = Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']), activation='relu')(x)
            else:
                x = Dense(param, input_shape=(input_dim, ), kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']), activation='relu')(model_input)
            #model.add(Dense(param, input_shape=(input_dim, ), kernel_initializer=keras.initializers.glorot_normal(seed=None), activation='relu'))
            i = 1
            if hyper_params['dropout'] > 0:
                x = Dropout(hyper_params['dropout'])(x)
        else:
            x = Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']), activation='relu')(x)
            #model.add(Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), activation='relu'))
            if hyper_params['dropout'] > 0:
                x = Dropout(hyper_params['dropout'])(x)
    # model.add(Dense(32, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='relu'))
    # model.add(Dropout(0.5))
    out2 = Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1_l2(hyper_params['l1'], hyper_params['l2']),activation='sigmoid', name='out2')(x)
    if hyper_params['trans_supervision']:
        model = Model(inputs=model_input, outputs=[out1, out2])
    else:
        model = Model(inputs=model_input, outputs=out2)

    model.summary()
    if hyper_params['save_iso']:
        model_iso = Model(inputs=model_input, outputs=out_map)
        model_iso.summary()
        return model, model_iso
    else:
        return model


def get_conv_model(input_dim):
    model_input = Input(shape=(input_dim, 1), name='Input')
    # filters, kernel_size
    filters = 32
    conv1 = Conv1D(filters, 128, strides=10, activation='relu', padding='same', name='conv1')(model_input)
    conv2 = Conv1D(filters, 128, strides=10, activation='relu', padding='same', name='conv2')(conv1)
    conv3 = Conv1D(filters, 64, strides=10, activation='relu', padding='same', name='conv3')(conv2)
    conv4 = Conv1D(filters, 64, strides=10, activation='relu', padding='same', name='conv4')(conv3)
    flatten = Flatten(name='flatten')(conv4)
    dense1 = Dense(64, activation='relu', name='dense1')(flatten)
    result = Dense(1, activation='sigmoid', name='dense_output')(dense1)
    model = Model(inputs=model_input, outputs=result)
    model.summary()
    return model

# split into input (X) and output (Y) variables
'''def get_model(input_dim, layer_params=[], hyper_params={}, dropout = False):
    # e.g. layer_params = 64, 32
    # create model
    model = Sequential()
    reg = keras.regularizers.l1(hyper_params['l1'])

    print(hyper_params['map_layer'])
    if hyper_params['map_layer']:
        print("hahaahahah")
        output_dim = hyper_params['linking_matrix'].shape[1]   
        model.add(MyMapLayer(output_dim, hyper_params['linking_matrix'], input_shape=(input_dim, )))
        # previously we add no activation
        model.add(Activation('relu'))

    if len(layer_params) == 0:
        # Here is just the logistic regression
        if hyper_params['map_layer']:
            model.add(Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='sigmoid'))
        else:
            model.add(Dense(1, input_shape=(input_dim, ), kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='sigmoid'))
        model.summary()
        return model

    i = 0
    for param in layer_params:
        if param <= 0:
            continue
        if (i == 0):
            if hyper_params['map_layer']:
                model.add(Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1(hyper_params['l1']), activation='relu'))
            else:
                model.add(Dense(param, input_shape=(input_dim, ), kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1(hyper_params['l1']), activation='relu'))
            #model.add(Dense(param, input_shape=(input_dim, ), kernel_initializer=keras.initializers.glorot_normal(seed=None), activation='relu'))
            i = 1
            if hyper_params['dropout'] > 0:
                model.add(Dropout(hyper_params['dropout']))
        else:
            model.add(Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=l1(hyper_params['l1']), activation='relu'))
            #model.add(Dense(param, kernel_initializer=keras.initializers.glorot_normal(seed=None), activation='relu'))
            if hyper_params['dropout'] > 0:
                model.add(Dropout(hyper_params['dropout']))
    # model.add(Dense(32, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation='sigmoid'))
    model.summary()
    return model

def get_conv_model(input_dim):
    model_input = Input(shape=(input_dim, 1), name='Input')
    # filters, kernel_size
    filters = 32
    conv1 = Conv1D(filters, 128, strides=10, activation='relu', padding='same', name='conv1')(model_input)
    conv2 = Conv1D(filters, 128, strides=10, activation='relu', padding='same', name='conv2')(conv1)
    conv3 = Conv1D(filters, 64, strides=10, activation='relu', padding='same', name='conv3')(conv2)
    conv4 = Conv1D(filters, 64, strides=10, activation='relu', padding='same', name='conv4')(conv3)
    flatten = Flatten(name='flatten')(conv4)
    dense1 = Dense(64, activation='relu', name='dense1')(flatten)
    result = Dense(1, activation='sigmoid', name='dense_output')(dense1)
    model = Model(inputs=model_input, outputs=result)
    model.summary()
    return model'''