import pickle
import gzip
import numpy as np
from keras.datasets import mnist
from models import create_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from scipy.special import softmax

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    ############
    # Parameters Section

    # the parameters for training the network
    epoch_num = 2
    batch_size = 500

    # Parameter C of SVM
    C = 1e-1
    # end of parameters section
    ############

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # set number of categories
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_category)
    y_test = to_categorical(y_test, num_category)

    # Building, training, and producing the new features by Deep LDA
    model = create_model(batch_size=batch_size)

    model_optimizer = Adam()
    callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.compile(loss=CategoricalCrossentropy(), optimizer=model_optimizer, metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, shuffle=True, validation_split=0.2, callbacks=[callback], verbose=2)

    print(history)

    print('History- ', history.history)

    # Training and testing the model
    train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
    test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Accuracy on train data is:", train_acc[1]*100)
    print("Accuracy on test data is:", test_acc[1]*100)

    # Saving model
    print('Saving model...')
    model.save("./model")


