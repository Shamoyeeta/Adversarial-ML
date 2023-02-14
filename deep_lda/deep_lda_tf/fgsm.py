import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import gzip
from objectives import lda_loss
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
import random
from svm import svm_classify

loss_object = lda_loss()

# the path to the final learned features
saved_parameters = './new_features.gz'

# the path to the saved model
model = tf.keras.models.load_model("./model", compile=False)
model.compile(loss=lda_loss(), optimizer=Adam())

# model = tf.keras.models.load_model("./model", custom_objects={'lda_loss': loss_object})

with gzip.open(saved_parameters, 'rb') as fp:
    lda_model_params = pickle.load(fp)

x_train_new = lda_model_params[0][0]
y_train_new = lda_model_params[0][1]

x_test_new = lda_model_params[1][0]
y_test_new = lda_model_params[1][1]


def create_adversarial_pattern(input_image, input_label):
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, [prediction])

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Get image and its label
# sample = random.sample(range(0, 10000), 10)
sample = 1  # random.randint(0, 1000)
image = [x_test[sample]]
label = [y_test[sample]]

perturbations = create_adversarial_pattern(image, label)
# visualize the perturbations
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    adv_x_new = model.predict(adv_x)
    [label_new, decision_vector] = svm_classify(x_train_new, y_train_new, adv_x_new, label)
    print("New prediction:", label_new)
    # print("Decision vector:", decision_vector)

    # plot the sample
    fig = plt.figure
    plt.imshow(adv_x, cmap='gray')
    plt.title('{} \n {} : {:.2f}% Confidence'.format(descriptions[i], label_new, np.argmax(decision_vector) * 100))
    plt.show()

# def img_plot():
#     num = 10
#     images = x_train[:num]
#     labels = y_train[:num]
#
#     num_row = 2
#     num_col = 5
#     # plot images
#     fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
#     for i in range(num):
#         ax = axes[i // num_col, i % num_col]
#         ax.imshow(images[i], cmap='gray')
#         ax.set_title('Label: {}'.format(labels[i]))
#     plt.tight_layout()
#     plt.show()
#
#
# img_plot()

# perturbations = create_adversarial_pattern(image, label)
# plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

# epsilons = [0, 0.01, 0.1, 0.15]
# descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
#                 for eps in epsilons]
#
# for i, eps in enumerate(epsilons):
#   adv_x = image + eps*perturbations
#   adv_x = tf.clip_by_value(adv_x, -1, 1)
#   display_images(adv_x, descriptions[i])
