import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import gzip
from objectives import lda_loss

# the path to save the final learned features
save_to = './new_features.gz'

with gzip.open(save_to, 'rb') as fp:
    lda_model_params = pickle.load(fp)

x_train = lda_model_params[0][0]
y_train = lda_model_params[0][1]

x_test = lda_model_params[1][0]
y_test = lda_model_params[1][1]

loss_object = lda_loss()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


