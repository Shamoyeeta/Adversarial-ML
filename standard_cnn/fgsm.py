import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import CategoricalCrossentropy
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
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
    plt.get_current_fig_manager().set_window_title("FGSM (epsilon= " + str(epsilon) + ")")
    plt.tight_layout()
    plt.show()




def create_adversarial_pattern(input_image, input_label):
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

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

perturbations = create_adversarial_pattern(image, label)
# # visualize the perturbations
# plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

epsilons = [0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = (image + (eps * perturbations)).numpy()
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    pred = np.argmax(model.predict(adv_x), axis=1)
    label = np.argmax(y_test[:20], axis=1)
    test_acc = accuracy_score(pred, label)
    print("New prediction on eps=" + str(eps) + " : ", test_acc*100)
    img_plot(adv_x[:10], eps, pred)
