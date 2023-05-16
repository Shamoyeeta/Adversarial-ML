"""
    Code for implementing JSMA attack on DeepLDA model using TensorflowV2
    Author: Vinoy Wilson
    Created: 15-04-2023
"""

import tensorflow as tf
import pickle
import gzip
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from objectives import lda_loss
from models import get_flatten_layer_output
from keras.datasets import mnist
from keras.optimizers import Adam
from svm import svm_classify
import time

tf.to_float = lambda x: tf.cast(x, tf.float32)

maxTime = 0
n_classes = 10


def jsma(model, x, y=None, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0,
         score_fn=lambda t, o: t * tf.abs(o)):
    """
    Jacobian-based saliency map approach.  During each iteration,
    this method finds the pixel (or two pixels) that has the most influence on
    the result (most salient pixel) and add noise to the pixel.
    :param model: A wrapper that returns the output tensor of the model.
    :param x: The input placeholder a 2D or 4D tensor.
    :param y: The desired class label for each input, either an integer or a
              list of integers.
    :param epochs: Maximum epochs to run.  If it is a floating number in [0,
        1], it is treated as the distortion factor, i.e., gamma in the
        original paper.
    :param eps: The noise added to input per epoch.
    :param k: number of pixels to perturb at a time.  Values other than 1 and
              2 will raise an error.
    :param clip_min: The minimum value in output tensor.
    :param clip_max: The maximum value in output tensor.
    :param score_fn: Function to calculate the saliency score.
    :return: A tensor, contains adversarial samples for each input.
    """
    n = tf.shape(x)[0]

    target = tf.cond(tf.equal(0, tf.rank(y)),
                     lambda: tf.zeros([n], dtype=tf.int32) + y,
                     lambda: y)
    target = tf.stack((tf.range(n), target), axis=1)

    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))

    if 2 == k:
        _jsma_fn = _jsma2_impl
    else:
        _jsma_fn = _jsma_impl

    return _jsma_fn(model, x, target, epochs=epochs, eps1=eps,
                    clip_min=clip_min, clip_max=clip_max, score_fn=score_fn)


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _jsma_impl(model, x, yind, epochs, eps1, clip_min, clip_max, score_fn):
    def _cond(i, xadv):
        return tf.less(i, epochs)

    def _body(i, xadv):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xadv)
            ybar = model(xadv)
            # gradients of target w.r.t input
            yt = tf.gather_nd(ybar, yind)

        dy_dx = tape.gradient(ybar, xadv)
        dt_dx = tape.gradient(yt, xadv)
        del tape

        # gradients of non-targets w.r.t input
        do_dx = dy_dx - dt_dx

        c0 = tf.logical_or(eps1 < 0, xadv < clip_max)
        c1 = tf.logical_or(eps1 > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        # cond = tf.cast(cond, tf.float32)
        cond = tf.to_float(cond)

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx)

        shape = score.get_shape().as_list()
        dim = _prod(shape[1:])
        score = tf.reshape(score, [-1, dim])

        # find the pixel with the highest saliency score
        ind = tf.argmax(score, axis=1)
        dx = tf.one_hot(ind, dim, on_value=eps1, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return i + 1, xadv

    _, xadv = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(_cond, _body, (0, tf.identity(x)),
                                                                    name='_jsma_batch'))

    return xadv


def _jsma2_impl(model, x, yind, epochs, eps1, clip_min, clip_max, score_fn):
    def _cond(k, xadv):
        return tf.less(k, epochs)

    def _body(k, xadv):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xadv)
            ybar = model(xadv)
            # gradients of target w.r.t input
            yt = tf.gather_nd(ybar, yind)

        dy_dx = tape.gradient(ybar, xadv)[0]
        dt_dx = tape.gradient(yt, xadv)[0]
        del tape

        # gradients of non-targets w.r.t input
        do_dx = dy_dx - dt_dx

        c0 = tf.logical_or(eps1 < 0, xadv < clip_max)
        c1 = tf.logical_or(eps1 > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx)

        shape = score.get_shape().as_list()
        dim = _prod(shape[1:])
        score = tf.reshape(score, [-1, dim])

        a = tf.expand_dims(score, axis=1)
        b = tf.expand_dims(score, axis=2)
        score2 = tf.reshape(a + b, [-1, dim * dim])
        ij = tf.argmax(score2, axis=1)

        i = tf.to_int32(ij / dim)
        j = tf.to_int32(ij) % dim

        dxi = tf.one_hot(i, dim, on_value=eps1, off_value=0.0)
        dxj = tf.one_hot(j, dim, on_value=eps1, off_value=0.0)
        dx = tf.reshape(dxi + dxj, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return k + 1, xadv

    _, xadv = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(_cond, _body, (0, tf.identity(x)),
                                                                    name='_jsma2_batch'))
    return xadv

def make_jsma(model, X_data, epochs=0.2, eps=1.0, batch_size=128):

    print('\nMaking adversarials via JSMA')
    global maxTime
    global n_classes

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        # feed_dict = {
        #     env.x: X_data[start:end],
        #     env.target: np.random.choice(n_classes),  # random choice of target class
        #     env.adv_epochs: epochs,
        #     env.adv_eps: eps}
        tick = time.perf_counter()
        # adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        adv = jsma(model, X_data[start:end], np.random.choice(n_classes), epochs=epochs, eps=eps)
        tock = time.perf_counter()
        maxTime = max(maxTime, (tock - tick))
        X_adv[start:end] = adv
    print("Maximum Time ", maxTime)

    return X_adv


def img_plot(images, epsilon, labels):
    num = images.shape[0]
    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title("Prediction = " + str(labels[i]))
    plt.get_current_fig_manager().set_window_title("JSMA (epsilon= " + str(epsilon) + ")")
    plt.tight_layout()
    plt.show()



loss_object = lda_loss()

# the path to the final learned features
saved_parameters = './new_features.gz'

# the path to the saved model
model = tf.keras.models.load_model("./model", compile=False)
model.compile(loss=lda_loss(), optimizer=Adam())

with gzip.open(saved_parameters, 'rb') as fp:
    lda_model_params = pickle.load(fp)

x_train_new = lda_model_params[0][0]
y_train_new = lda_model_params[0][1]

x_test_new = lda_model_params[1][0]
y_test_new = lda_model_params[1][1]

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# set number of categories
num_category = 10

image = x_test
label = y_test

print('\nEvaluating on original data')
[train_acc, test_acc, pred] = svm_classify(x_train_new, y_train_new , x_test_new , y_test_new )
print("Prediction on original data= ", test_acc * 100)

epsilons = [0.0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]

for i, eps in enumerate(epsilons):
    print('\nGenerating adversarial data') \
        # X_adv = make_cw(sess, env, X_test, epochs=30, eps=3)
    X_adv = make_jsma(model, image, epochs=30, eps=eps)

    print('\nEvaluating on adversarial data')
    X_adv_new = get_flatten_layer_output(model, X_adv)

    [train_acc, test_acc, pred] = svm_classify(x_train_new, y_train_new , X_adv_new, label)

    print("Prediction on adversarial data (eps = " + str(eps) + ")= ", test_acc * 100)
    img_plot(X_adv[:10], eps, pred)