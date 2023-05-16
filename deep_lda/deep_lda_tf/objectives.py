"""
    Code for LDA loss function using TensorflowV2
    Author: Shamoyeeta Saha
    Created: 08-02-2023
"""

import tensorflow as tf
from tensorflow.python.keras.layers import dot
import numpy as np
import keras
import scipy.linalg as slinalg


def numpy_unique(a):
    return np.unique(a)

@tf.custom_gradient
def eigvalsh(A, B):
    """ Solving the generalized eigenvalue problem A x = lambda B x

    Gradients of this function is customized.

    Parameters
    ----------
    A: tf.Tensor
        Left-side matrix with shape [D, D]
    B: tf.Tensor
        Right-side matrix with shape [D, D]

    Returns
    -------
    w: tf.Tensor
        Eigenvalues, with shape [D]
    grad: function
        Gradient of this function

    Reference:
    https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/slinalg.py#L385-L440

    """
    w, v = tf.py_function(slinalg.eigh, inp=[A, B], Tout=[tf.float32, tf.float32])
    w.set_shape(A.shape[0])  # set_shape here is necessary

    def grad(dw):
        gA = tf.matmul(v, tf.matmul(tf.linalg.diag(dw), tf.transpose(v, (1, 0))))
        gB = -tf.matmul(v, tf.matmul(tf.linalg.diag(dw * w), tf.transpose(v, (1, 0))))

        # The two steps below seem no effect on the final computed gradients
        # Uncomment these lines if needed.

        # gA = tf.linalg.band_part(gA, -1, 0) \
        #     + tf.transpose(tf.linalg.band_part(gA, 0, -1), perm=(0, 2, 1)) \
        #     - tf.linalg.band_part(gA, 0, 0)
        # gB = tf.linalg.band_part(gB, -1, 0) \
        #     + tf.transpose(tf.linalg.band_part(gB, 0, -1), perm=(0, 2, 1)) \
        #     - tf.linalg.band_part(gB, 0, 0)
        return [gA, gB]

    return w, grad


def eigh(A, B):
    return tf.py_function(slinalg.eigh, inp=[A, B], Tout=[tf.float32, tf.float32])


def linear_discriminative_eigvals(y, X, lambda_val=1e-3, ret_vecs=False):
    """
    Compute the linear discriminative eigenvalues

    Usage:

    >>> y = [0, 0, 1, 1]
    >>> X = [[1, -2], [-3, 2], [1, 1.4], [-3.5, 1]]
    >>> eigvals = linear_discriminative_eigvals(y, X, 2)
    >>> eigvals.numpy()
    [-0.33328852 -0.17815116]

    Parameters
    ----------
    y: tf.Tensor, np.ndarray
        Ground truth values, with shape [N, 1]
    X: tf.Tensor, np.ndarray
        The predicted values (i.e., features), with shape [N, d].
    lambda_val: float
        Lambda for stablizing the right-side matrix of the generalized eigenvalue problem
    ret_vecs: bool
        Return eigenvectors or not.
        **Notice:** If False, only eigenvalues are returned and this function supports
        backpropagation (used for training); If True, both eigenvalues and eigenvectors
        are returned but the backpropagation is undefined (used for validation).

    Returns
    -------
    eigvals: tf.Tensor
        Linear discriminative eigenvalues, with shape [cls]

    References:
    Dorfer M, Kelz R, Widmer G. Deep linear discriminant analysis[J]. arXiv preprint arXiv:1511.04707, 2015.

    """
    X = tf.convert_to_tensor(X, tf.float32)  # [N, d]
    y = tf.squeeze(tf.cast(tf.convert_to_tensor(y), tf.int32))  # [N]
    y.set_shape(X.shape[:-1])  # [N]
    classes = tf.sort(tf.raw_ops.UniqueV2(x=y, axis=[0]).y)
    num_classes = tf.shape(classes)[0]

    def compute_cov(args):
        i, Xcopy, ycopy = args
        # Hypothesis: equal number of samples (Ni) for each class
        Xg = Xcopy[ycopy == i]  # [None, d]
        Xg_bar = Xg - tf.reduce_mean(Xg, axis=0, keepdims=True)  # [None, d]
        m = tf.cast(tf.shape(Xg_bar)[0], tf.float32)  # []
        Xg_bar_dummy_batch = tf.expand_dims(Xg_bar, axis=0)  # [1, None, d]
        ans = (1. / (m - 1)) * tf.squeeze(
            dot([Xg_bar_dummy_batch, Xg_bar_dummy_batch], axes=1), axis=0)  # [d, d]
        try:
            ans = np.nan_to_num(ans)
        except:
            print("ans -", ans)
        return ans

    # covariance matrixs for all the classes
    covs_t = tf.map_fn(
        compute_cov, (classes,
                      tf.repeat(tf.expand_dims(X, 0), num_classes, axis=0),
                      tf.repeat(tf.expand_dims(y, 0), num_classes, axis=0)),
        fn_output_signature=tf.float32)  # [cls, d, d]
    # Within-class scatter matrix
    Sw = tf.reduce_mean(covs_t, axis=0)  # [d, d]

    # Total scatter matrix
    X_bar = X - tf.reduce_mean(X, axis=0, keepdims=True)  # [N, d]
    m = tf.cast(X_bar.shape[0], tf.float32)  # []
    X_bar_dummy_batch = tf.expand_dims(X_bar, axis=0)  # [1, N, d]
    St = (1. / (m - 1)) * tf.squeeze(
        dot([X_bar_dummy_batch, X_bar_dummy_batch], axes=1), axis=0)  # [d, d]

    # Between-class scatter matrix
    Sb = St - Sw  # [d, d]

    # Force Sw_t to be positive-definite (for numerical stability)
    Sw = Sw + tf.eye(Sw.shape[0]) * lambda_val  # [d, d]

    # Solve the generalized eigenvalue problem: Sb * W = lambda * Sw * W
    # We use the customed `eigh` function for generalized eigenvalue problem
    if ret_vecs:
        return eigh(Sb, Sw)  # [cls], [d, cls]
    else:
        return eigvalsh(Sb, Sw)  # [cls]


class lda_loss(keras.losses.Loss):
    def __init__(self, name="lda_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        eigvals = linear_discriminative_eigvals(y_true, y_pred)  # [cls]

        # At most cls - 1 non-zero eigenvalues
        classes = tf.raw_ops.UniqueV2(x=y_true, axis=[0]).y  # [cls]
        cls = tf.shape(classes)[0]
        eigvals = eigvals[-cls + 1:]  # [cls - 1]
        thresh = tf.reduce_min(eigvals) + 1.0  # []

        # maximize variance between classes
        top_k_eigvals = eigvals[eigvals <= thresh]  # [None]
        costs = -tf.reduce_mean(top_k_eigvals)  # []

        return costs
