"""
    Code for implementing Carlini and Wagner's attack on model using TensorflowV2
    Author: Shamoyeeta Saha
    Created: 23-04-2023
"""

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

def inv_softmax(x, C):
   return tf.math.log(x) + C


def cw(model, noise, x, y=None, tau=None, eps=1.0, ord_=2, T=2,
       optimizer=Adam(learning_rate=0.1), alpha=0.9,
       min_prob=0.5, clip=(0.0, 1.0)):
    """CarliniWagner (CW) attack.
    Only CW-L2 and CW-Linf are implemented since I do not see the point of
    embedding CW-L2 in CW-L1.  See https://arxiv.org/abs/1608.04644 for
    details.
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
    # # print('tau before-', tau)
    # print('Noise begin -',noise)

    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1 - 1e-8)
    xinv = tf.math.log(z / (1 - z)) / T

    get_logit_layer_output = K.function(
        [model.layers[0].input],  # param 1 will be treated as layer[0].output
        [model.get_layer('dense').output])  # and this function will return output from flatten layer

    with tf.GradientTape() as tape:

        # add noise in sigmoid-space and map back to input domain
        xadv = tf.sigmoid(T * (xinv + noise))  # 1
        xadv = xadv * (clip[1] - clip[0]) + clip[0]  # 2

        # ybar, logits = model(xadv, logits=True)
        # logits = get_logit_layer_output(xadv)[0]

        # print(logits)
        ybar = model(xadv)
        logits = inv_softmax(ybar, tf.math.log(10.))
        # print(ybar)
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

    # print('Noise before opt', tf.reduce_sum(abs(noise)))
    train_op = optimizer.minimize(loss, var_list=[noise], tape=tape)
    # print('Noise after opt', tf.reduce_sum(abs(noise)))

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        # add noise in sigmoid-space and map back to input domain
        xadv = tf.sigmoid(T * (xinv + noise))  # 1
        xadv = xadv * (clip[1] - clip[0]) + clip[0]  # 2
        diff = xadv - x - tau
        tau = alpha * tf.cast(tf.reduce_all(input_tensor=diff < 0, axis=axis), dtype=tf.float32)

    # # print('tau after-', tau)
    # print('noise end -', noise)
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
            # print('tau before 1-',tau)
            for epoch in range(epochs):
                # env.sess.run(env.adv_train_op, feed_dict=feed_dict)
                adv_train_op, xadv, noise, tau = cw(model, noise, x,
                                                    y=5, tau=tau, eps=eps, ord_=2, clip=clip)

            # xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            adv_train_op, xadv, noise, tau = cw(model, noise, x,
                                                y=5, tau=tau, eps=eps, ord_=2)

            # print('tau after-', tau)
            # print('Diff - ', tf.reduce_sum(xadv-X_data))
            X_adv[start:end] = xadv
            Noise[start:end] = noise

    # print('Data - ', X_data)
    # print('Adv - ',X_adv)
    # print('Noise-', Noise)
    return X_adv

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
image = x_test
label = y_test

epsilons = [0.0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 1, 3]

for i, eps in enumerate(epsilons):
  print('\nGenerating adversarial data')
  # X_adv = make_cw(sess, env, X_test, epochs=30, eps=3)
  X_adv = make_cw(model, image, epochs=100, eps=eps)
  print("Diff val abs -", tf.reduce_sum(abs(X_adv) - abs(image)))
  print(model.predict(X_adv))

  print('\nEvaluating on adversarial data')
  pred = np.argmax(model.predict(X_adv), axis=1)
  label = np.argmax(y_test, axis=1)
  test_acc = accuracy_score(pred, label)

  print("Prediction on adversarial data (eps = " + str(eps)+")= ", test_acc * 100)
  img_plot(X_adv[:10], eps, pred)