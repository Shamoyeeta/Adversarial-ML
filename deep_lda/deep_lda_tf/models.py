# from keras.layers import Dense
# from keras.models import Sequential
# from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np


def create_model(batch_size=400):
    """
    Builds the model
    The structure of the model can get easily substituted with a more efficient and powerful network like CNN
    """

    l_in = keras.Input(shape=(28, 28, 1), batch_size=batch_size)
    print(l_in.shape)

    # ---conv layer ---
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(l_in)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.MaxPool2D()(net)
    net = tf.keras.layers.Dropout(rate=0.25)(net)

    net = tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.MaxPool2D()(net)
    net = tf.keras.layers.Dropout(rate=0.25)(net)

    net = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(rate=0.5)(net)
    net = tf.keras.layers.Conv2D(filters=256, kernel_size=2, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(rate=0.5)(net)

    net = tf.keras.layers.Conv2D(filters=10, kernel_size=1, activation='relu',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.Flatten()(net)
    l_out = tf.keras.layers.Dense(10)(net)

    model_built = tf.keras.Model(l_in, l_out)
    print(model_built.summary())
    return model_built


def get_flatten_layer_output(model, x, batch_size=400):
    flatten_layer_output = K.function(
        [model.layers[0].input],  # param 1 will be treated as layer[0].output
        [model.get_layer('flatten').output])  # and this function will return output from flatten layer

    n_sample = x.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    flatten_output = np.empty([n_sample, 10])

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feature_vector = flatten_layer_output(x[start:end])
        flatten_output[start:end] = feature_vector[0]
    return flatten_output
