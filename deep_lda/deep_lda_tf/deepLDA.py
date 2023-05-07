import pickle
import gzip
import numpy as np
from keras.datasets import mnist
from svm import svm_classify
from models import create_model, get_flatten_layer_output
from keras.optimizers import Adam
from objectives import lda_loss
from keras.callbacks import EarlyStopping

import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    ############
    # Parameters Section

    # the path to save the final learned features
    save_features = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10

    # the parameters for training the network
    epoch_num = 2 #100
    batch_size = 500

    # The margin and n_components (number of components) parameter used in the loss function
    # n_components should be at most class_size-1
    margin = 1.0
    n_components = 9

    # Parameter C of SVM
    C = 1e-1
    # end of parameters section
    ############

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Building, training, and producing the new features by Deep LDA
    model = create_model(batch_size=batch_size)

    model_optimizer = Adam()
    callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.compile(loss=lda_loss(), optimizer=model_optimizer)

    model.summary()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, shuffle=True, validation_split=0.2, callbacks=[callback], verbose=2)

    print('History- ', history.history)

    x_train_new = get_flatten_layer_output(model, x_train)
    x_test_new = get_flatten_layer_output(model, x_test)

    # Saving model parameters in a gzip pickled file specified by save_model
    print('Saving model...')
    model.save("./model")

    # Training and testing of SVM with linear kernel on the new features
    [train_acc, test_acc, pred] = svm_classify(x_train_new, y_train, x_test_new, y_test, C=C)
    print("Accuracy on train data is:", train_acc * 100.0)
    print("Accuracy on test data is:", test_acc*100.0)

    # Saving new features in a gzip pickled file specified by save_features
    print('Saving new features ...')
    f = gzip.open(save_features, 'wb')
    pickle.dump([(x_train_new, y_train), (x_test_new, y_test)], f)
    f.close()

