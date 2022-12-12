from models.mnist_dlda import objective_tf
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

objective_tf(tf.zeros((3, 3)),  tf.zeros((3, 3)))