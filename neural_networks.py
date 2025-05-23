import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, Sequential

def build_standard_nn(hidden_layer, neurons_per_layer, activation_function, weight_initializer):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,), name="input"))
    for i in range(hidden_layer):
        model.add(tf.keras.layers.Dense(
            units=neurons_per_layer,
            activation=activation_function,
            kernel_initializer=weight_initializer(),
            name=f"hidden_{i}"))
    model.add(tf.keras.layers.Dense(units=3, activation=None, kernel_initializer=weight_initializer(), name="output"))
    return model

def build_l2_regularized_nn(hidden_layer, neurons_per_layer, activation_function, weight_initializer, l2_factor=0.1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,), name="input"))
    for i in range(hidden_layer):
        model.add(tf.keras.layers.Dense(
            units=neurons_per_layer,
            activation=activation_function,
            #kernel_initializer=weight_initializer(), // NOTE: UNCOMMENT THIS LINE IF IT ISN'T -DEBUG PURPOSES
            kernel_regularizer=regularizers.l2(l2_factor),
            name=f"hidden_{i}"))
    model.add(tf.keras.layers.Dense(units=3, activation=None, kernel_initializer=weight_initializer(), name="output"))
    return model
