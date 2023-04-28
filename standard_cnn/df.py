import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import CategoricalCrossentropy
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time
import os
from timeit import default_timer
import matplotlib.pyplot as plt
from keras import backend as K

maxTime = 0


def deepfool(model, x, noise=False, eta=0.02, epochs=3, batch=False,
             clip_min=0.0, clip_max=1.0, min_prob=0.0):
    """DeepFool implementation in Tensorflow.
    The original DeepFool will stop whenever we successfully cross the
    decision boundary.  Thus it might not run total epochs.  In order to force
    DeepFool to run full epochs, you could set batch=True.  In that case the
    DeepFool will run until the max epochs is reached regardless whether we
    cross the boundary or not.  See https://arxiv.org/abs/1511.04599 for
    details.
    :param model: Model function.
    :param x: 2D or 4D input tensor.
    :param noise: Also return the noise if True.
    :param eta: Small overshoot value to cross the boundary.
    :param epochs: Maximum epochs to run.
    :param batch: If True, run in batch mode, will always run epochs.
    :param clip_min: Min clip value for output.
    :param clip_max: Max clip value for output.
    :param min_prob: Minimum probability for adversarial samples.
    :return: Adversarials, of the same shape as x.
    """
    y = tf.stop_gradient(model(x))

    fns = [[_deepfool2, _deepfool2_batch], [_deepfoolx, _deepfoolx_batch]]

    i = int(y.get_shape().as_list()[1] > 1)
    j = int(batch)
    fn = fns[i][j]

    if batch:
        delta = fn(model, x, eta=eta, epochs=epochs, clip_min=clip_min,
                   clip_max=clip_max)
    else:
        def _f(xi):
            xi = tf.expand_dims(xi, axis=0)
            z = fn(model, xi, eta=eta, epochs=epochs, clip_min=clip_min,
                   clip_max=clip_max, min_prob=min_prob)
            return z[0]

        # delta = tf.map_fn(_f, x, dtype=(tf.float32), back_prop=False,
        #                   name='deepfool')
        delta = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(_f, x, dtype=(tf.float32), name='deepfool'))

    if noise:
        print('Noise - ', delta)
        return delta

    xadv = tf.stop_gradient(x + delta * (1 + eta))
    # print('In function - ', xadv)
    xadv = tf.clip_by_value(xadv, clip_min, clip_max)
    return xadv


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _deepfool2(model, x, epochs, eta, clip_min, clip_max, min_prob):
    """DeepFool for binary classifiers.
    Note that DeepFools that binary classifier outputs +1/-1 instead of 0/1.
    """
    y0 = tf.stop_gradient(tf.reshape(model(x), [-1])[0])
    y0 = tf.cast(tf.greater(y0, 0.0), dtype=tf.int32)

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        y = tf.stop_gradient(tf.reshape(model(xadv), [-1])[0])
        y = tf.cast(tf.greater(y, 0.0), dtype=tf.int32)
        return tf.logical_and(tf.less(i, epochs), tf.equal(y0, y))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        with tf.GradientTape as tape:
            tape.watch(xadv)
            y = tf.reshape(model(xadv), [-1])[0]
        g = tape.gradient(y, xadv)
        dx = - y * g / (tf.norm(tensor=g) + 1e-10)  # off by a factor of 1/norm(g)
        return i + 1, z + dx

    # _, noise = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
    #                          name='_deepfool2', back_prop=False)
    _, noise = tf.nest.map_structure(tf.stop_gradient,
                                     tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
                                                   name='_deepfool2'))
    return noise


def _deepfool2_batch(model, x, epochs, eta, clip_min, clip_max):
    """DeepFool for binary classifiers in batch mode.
    """
    xshape = x.get_shape().as_list()[1:]
    dim = _prod(xshape)

    def _cond(i, z):
        return tf.less(i, epochs)

    def _body(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        with tf.GradientTape as tape:
            tape.watch(xadv)
            y = tf.reshape(model(xadv), [-1])[0]
        g = tape.gradient(y, xadv)
        n = tf.norm(tensor=tf.reshape(g, [-1, dim]), axis=1) + 1e-10
        d = tf.reshape(-y / n, [-1] + [1] * len(xshape))
        dx = g * d
        return i + 1, z + dx

    # _, noise = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
    #                          name='_deepfool2_batch', back_prop=False)
    _, noise = tf.nest.map_structure(tf.stop_gradient,
                                     tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
                                                   name='_deepfool2_batch'))
    return noise


def _deepfoolx(model, x, epochs, eta, clip_min, clip_max, min_prob):
    """DeepFool for multi-class classifiers.
    Assumes that the final label is the label with the maximum values.
    """
    y0 = tf.stop_gradient(model(x))
    y0 = tf.reshape(y0, [-1])
    k0 = tf.argmax(input=y0)

    ydim = y0.get_shape().as_list()[0]
    xdim = x.get_shape().as_list()[1:]
    xflat = _prod(xdim)

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])
        p = tf.reduce_max(input_tensor=y)
        k = tf.argmax(input=y)
        return tf.logical_and(tf.less(i, epochs),
                              tf.logical_or(tf.equal(k0, k),
                                            tf.less(p, min_prob)))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xadv)
            y = model(xadv)

        # print(ydim)

        gs = [tf.reshape(tape.gradient(y, xadv), [-1])
              for i in range(ydim)]
        del tape
        g = tf.stack(gs, axis=0)
        y = tf.reshape(y, [-1])
        # print(y)
        # print('gs -', gs)
        # print('g - ', g)

        yk, yo = y[k0], tf.concat((y[:k0], y[(k0 + 1):]), axis=0)
        gk, go = g[k0], tf.concat((g[:k0], g[(k0 + 1):]), axis=0)

        yo.set_shape(ydim - 1)
        go.set_shape([ydim - 1, xflat])

        a = tf.abs(yo - yk)
        b = go - gk
        c = tf.norm(tensor=b, axis=1)
        score = a / c
        ind = tf.argmin(input=score)

        si, bi = score[ind], b[ind]
        dx = si * bi
        dx = tf.reshape(dx, [-1] + xdim)
        return i + 1, z + dx

    # _, noise = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
    #                          name='_deepfoolx', back_prop=False)
    _, noise = tf.nest.map_structure(tf.stop_gradient,
                                     tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
                                                   name='_deepfoolx'))
    # print('Noise from deepfool', noise)
    return noise


def _deepfoolx_batch(model, x, epochs, eta, clip_min, clip_max):
    """DeepFool for multi-class classifiers in batch mode.
    """
    y0 = tf.stop_gradient(model(x))
    B, ydim = tf.shape(input=y0)[0], y0.get_shape().as_list()[1]

    k0 = tf.argmax(input=y0, axis=1, output_type=tf.int32)
    k0 = tf.stack((tf.range(B), k0), axis=1)

    xshape = x.get_shape().as_list()[1:]
    xdim = _prod(xshape)

    perm = list(range(len(xshape) + 2))
    perm[0], perm[1] = perm[1], perm[0]

    def _cond(i, z):
        return tf.less(i, epochs)

    def _body(i, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        y = model(xadv)

        h = tf.GradientTape(ys=y[:, i], xs=xadv)[0]
        gs = [h for i in range(ydim)]
        g = tf.stack(gs, axis=0)
        g = tf.transpose(a=g, perm=perm)

        yk = tf.expand_dims(tf.gather_nd(y, k0), axis=1)
        gk = tf.expand_dims(tf.gather_nd(g, k0), axis=1)

        a = tf.abs(y - yk)
        b = g - gk
        c = tf.norm(tensor=tf.reshape(b, [-1, ydim, xdim]), axis=-1)

        # Assume 1) 0/0=tf.nan 2) tf.argmin ignores nan
        score = a / c

        ind = tf.argmin(input=score, axis=1, output_type=tf.int32)
        ind = tf.stack((tf.range(B), ind), axis=1)

        si, bi = tf.gather_nd(score, ind), tf.gather_nd(b, ind)
        si = tf.reshape(si, [-1] + [1] * len(xshape))
        dx = si * bi
        return i + 1, z + dx

    # _, noise = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
    #                          name='_deepfoolx_batch', back_prop=False)
    _, noise = tf.nest.map_structure(tf.stop_gradient,
                                     tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
                                                   name='_deepfoolx_batch'))
    return noise


def make_deepfool(model, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nMaking adversarials via DeepFool')
    global maxTime

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tick = time.perf_counter()
        # adv = sess.run(env.xadv, feed_dict={env.x: X_data[start:end],
        #         #                                     env.adv_epochs: epochs})
        adv = deepfool(model, X_data[start:end], epochs=epochs)
        tock = time.perf_counter()
        maxTime = max(maxTime, (tock - tick))
        X_adv[start:end] = adv
    print("Maximum Time ", maxTime)

    return X_adv


def img_plot(images, labels):
    num = images.shape[0]
    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title("Prediction = " + str(labels[i]))
    plt.get_current_fig_manager().set_window_title("Deepfool")
    plt.tight_layout()
    plt.show()

def img_plot(images, labels):
    num = images.shape[0]
    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title("Prediction = " + str(labels[i]))
    plt.get_current_fig_manager().set_window_title("Deepfool")
    plt.tight_layout()
    plt.show()



# the path to the saved model
model = tf.keras.models.load_model("./model", compile=False)
model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"])
loss_object = CategoricalCrossentropy()

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

# Get image and its label
image = x_test[:20]
label = y_test[:20]

print('\nGenerating adversarial data')
X_adv = make_deepfool(model, image, epochs=1)

print('\nEvaluating on adversarial data')
print(model(X_adv))
pred = np.argmax(model.predict(X_adv), axis=1)
label = np.argmax(y_test[:20], axis=1)
test_acc = accuracy_score(pred, label)
print(pred)

print("Prediction on adversarial data= ", test_acc * 100)
img_plot(X_adv[:10], pred)
