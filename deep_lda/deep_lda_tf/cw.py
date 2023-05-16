"""
    Code for implementing Carlini and Wagner's attack on DeepLDA model using TensorflowV2
    Author: Shamoyeeta Saha
    Created: 03-04-2023
"""

import os
from timeit import default_timer
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import gzip
from keras.datasets import mnist
from keras.optimizers import Adam

from objectives import lda_loss
from svm import svm_classify
from models import get_flatten_layer_output

def inv_softmax(x, C):
   return tf.math.log(x) + C


def cw(model, noise, x, y=None, tau=None, eps=1.0, ord_=2, T=2,
       optimizer=Adam(learning_rate=0.1), alpha=0.9,
       min_prob=0, clip=(0.0, 1.0)):
    """CarliniWagner (CW) attack.
    Only CW-L2 and CW-Linf are implemented
    The idea of CW attack is to minimize a loss that comprises two parts: a)
    the p-norm distance between the original image and the adversarial image,
    and b) a term that encourages the incorrect classification of the
    adversarial images.
    Please note that CW is a optimization process, so it is tricky.  There are
    lots of hyper-parameters to tune in order to get the best result.  The
    binary search process for the best eps values is omitted here.  You could
    do grid search to find the best parameter configuration, if you like.  I
    demonstrate binary search for the best result in an example code.
    :param model: The model wrapper.
    :param x: The input clean sample, usually a placeholder.  NOTE that the
              shape of x MUST be static, i.e., fixed when constructing the
              graph.  This is because there are some variables that depends
              upon this shape.
    :param y: The target label.  Set to be the least-likely label when None.
    :param eps: The scaling factor for the second penalty term.
    :param ord_: The p-norm, 2 or inf.  Actually I only test whether it is 2
        or not 2.
    :param T: The temperature for sigmoid function.  In the original paper,
              the author used (tanh(x)+1)/2 = sigmoid(2x), i.e., t=2.  During
              our experiment, we found that this parameter also affects the
              quality of generated adversarial samples.
    :param optimizer: The optimizer used to minimize the CW loss.  Default to
        be tf.AdamOptimizer with learning rate 0.1. Note the learning rate is
        much larger than normal learning rate.
    :param alpha: Used only in CW-L0.  The decreasing factor for the upper
        bound of noise.
    :param min_prob: The minimum confidence of adversarial examples.
        Generally larger min_prob will result in more noise.
    :param clip: A tuple (clip_min, clip_max), which denotes the range of
        values in x.
    :return: A tuple (train_op, xadv, noise).  Run train_op for some epochs to
             generate the adversarial image, then run xadv to get the final
             adversarial image.  Noise is in the sigmoid-space instead of the
             input space.  It is returned because we need to clear noise
             before each batched attacks.
    """
    xshape = x.get_shape().as_list()
    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)

    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1 - 1e-8)
    xinv = tf.math.log(z / (1 - z)) / T

    with tf.GradientTape() as tape:

        # add noise in sigmoid-space and map back to input domain
        xadv = tf.sigmoid(T * (xinv + noise))  # 1
        xadv = xadv * (clip[1] - clip[0]) + clip[0]  # 2

        ybar = model(xadv)
        logits = inv_softmax(ybar, tf.math.log(10.))

        ydim = ybar.shape[1]

        if y is not None:
            y = tf.cond(pred=tf.equal(tf.rank(y), 0),
                        true_fn=lambda: tf.fill([xshape[0]], y),
                        false_fn=lambda: tf.identity(y))
        else:
            # we set target to the least-likely label
            y = tf.argmin(input=ybar, axis=1, output_type=tf.int32)

        # print('y -', y)

        mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
        yt = tf.reduce_max(input_tensor=logits - mask, axis=1)
        yo = tf.reduce_max(input_tensor=logits, axis=1)

        # encourage to classify to a wrong category
        loss0 = tf.nn.relu(yo - yt + min_prob)

        # make sure the adversarial images are visually close
        if 2 == ord_:
            # CW-L2 Original paper uses the reduce_sum version.  These two
            # implementation does not differ much.

            # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
            loss1 = tf.reduce_mean(input_tensor=tf.square(xadv - x))
        else:
            # CW-Linf

            diff = xadv - x - tau

            # if all values are smaller than the upper bound value tau, we reduce
            # this value via tau*0.9 to make sure L-inf does not get stuck.
            loss1 = tf.nn.relu(tf.reduce_sum(input_tensor=diff, axis=axis))

        loss = eps * loss0 + loss1

    train_op = optimizer.minimize(loss, var_list=[noise], tape=tape)

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        # add noise in sigmoid-space and map back to input domain
        xadv1 = tf.sigmoid(T * (xinv + noise))  # 1
        xadv1 = xadv1 * (clip[1] - clip[0]) + clip[0]  # 2
        diff = xadv1 - x - tau
        tau = alpha * tf.cast(tf.reduce_all(input_tensor=diff < 0, axis=axis), dtype=tf.float32)

    return train_op, xadv, noise, tau


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_size = 28
img_chan = 1
n_classes = 10
batch_size = 500


class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        print(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        print(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


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
    plt.get_current_fig_manager().set_window_title("CW (epsilon= " + str(epsilon) + ")")
    plt.tight_layout()
    plt.show()


def make_cw(model, X_data, epochs=1, eps=1, batch_size=batch_size):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    Noise = np.empty_like(X_data)

    # for batch in range(n_batch):
    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch + 1) * batch_size)
            start = end - batch_size

            xshape = X_data[start:end].shape
            axis = list(range(1, len(xshape)))
            clip = (0.0, 1.0)
            tau0 = tf.fill([xshape[0]] + [1] * len(axis), clip[1])

            tau = tf.Variable(tau0, trainable=False, dtype=tf.float32, name='cw8-noise-upperbound')
            noise = tf.Variable(tf.zeros(xshape, dtype=tf.float32), dtype=tf.float32, name='noise', trainable=True)
            x = tf.convert_to_tensor(X_data[start:end])
            for epoch in range(epochs):
                adv_train_op, xadv, noise, tau = cw(model, noise, x,
                                                    y=5, tau=tau, eps=eps, ord_=2, clip=clip)

            adv_train_op, xadv, noise, tau = cw(model, noise, x,
                                                y=5, tau=tau, eps=eps, ord_=2)

            X_adv[start:end] = xadv
            Noise[start:end] = noise

    return X_adv


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

epsilons = [0.0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 1, 3]

print('\nEvaluating on original data')
[train_acc, test_acc, pred] = svm_classify(x_train_new, y_train_new, x_test_new, y_test_new)
print("Prediction on original data= ", test_acc * 100)

for i, eps in enumerate(epsilons):
    print('\nGenerating adversarial data')
    X_adv = make_cw(model, image, epochs=100, eps=eps)

    print('\nEvaluating on adversarial data')
    X_adv_new = get_flatten_layer_output(model, X_adv)

    [train_acc, test_acc, pred] = svm_classify(x_train_new, y_train_new, X_adv_new, label)

    print("Prediction on adversarial data (eps = " + str(eps) + ")= ", test_acc * 100)
    img_plot(X_adv[:10], eps, pred)
