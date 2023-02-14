# from keras.layers import Dense
# from keras.models import Sequential
# from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras

BATCH_SIZE = 1000

def create_model(batch_size = BATCH_SIZE):
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
    l_out = tf.keras.layers.Flatten()(net)

    model_built = tf.keras.Model(l_in, l_out)
    print(model_built.summary())
    return model_built
