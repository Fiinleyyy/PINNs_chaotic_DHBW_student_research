import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, Sequential, callbacks

def build_standard_nn(hidden_layer, neurons_per_layer, activation_function, weight_initializer):
    """Builds a standard neuronal network."""
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
    """Builds a l2-regulated standard neuronal network."""
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


class LossStatusCallback(callbacks.Callback):
    """Creates loss status callbacks."""
    def __init__(self, number_epochs):
        super().__init__()
        self.number_epochs = number_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.number_epochs == 0:
            loss = logs.get("loss")
            print(f"Epoch {epoch + 1}: loss = {loss:.5f}")