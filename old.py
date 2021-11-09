import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
from matplotlib import pyplot as plt
from PIL import Image

# The dimensions of our input image
img_width = 512
img_height = 512
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "Conv2DT_1"

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display1(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    
    plt.savefig('out.png')
    plt.show()


def compute_loss(input_image, filter_index, feature_extractor):
    """
    Computes the loss values for a specific image
    """
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    """
    Performs a step of gradient ascent
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image():
    """
    Initializes an image with random noise
    """
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 1))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index, feature_extractor):
    """
    Visualize filter a specific filter in a convolutional layer
    """
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    print(img)
    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    """
    Deprocess image
    """
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    #img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def load_data(data_dir):
    """
    Load the raw image data into a data matrix, X, and target vector, y.

    :param data_dir: The root directory of the data
    :return: the data matrix 'x' and target vector 'y'
    """
    x = []
    y = []
    for file_name in os.listdir(data_dir + 'train/'):
        image_file = os.path.join(data_dir + 'train/', file_name)
        im = np.array(load_img(image_file))
        im = tf.image.resize(im, size=(512, 512)).numpy()
        im = tf.image.rgb_to_grayscale(im)
        x.append(im)
    for file_name in os.listdir(data_dir + 'test/'):
        image_file = os.path.join(data_dir + 'test/', file_name)
        im = np.array(load_img(image_file))
        im = tf.image.resize(im, size=(512, 512)).numpy()
        im = tf.image.rgb_to_grayscale(im)
        y.append(im)
    x = np.array(x)
    y = np.array(y)
    return x, y



def main():

    # Since we only need images from the dataset to encode and decode, we
    # won't use the labels.
    train_data, test_data = load_data('data/')

    input = layers.Input(shape=(512, 512, 1))

    # Encoder
    x = layers.Conv2D(32, (16, 16), activation="relu", padding="same", name='Conv2D_1')(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (16, 16), activation="relu", padding="same", name='Conv2D_2')(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (16, 16), activation="relu", padding="same", name='Conv2D_3')(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (16, 16), activation="relu", padding="same", name='Conv2D_4')(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Can't go smaller or else we can't reconstruct back into the correct image size.

    # Decoder
    x = layers.Conv2DTranspose(32, (16, 16), strides=2, activation="relu", padding="same", name='Conv2DT_1')(x)
    x = layers.Conv2DTranspose(32, (16, 16), strides=2, activation="relu", padding="same", name='Conv2DT_2')(x)
    x = layers.Conv2DTranspose(32, (16, 16), strides=2, activation="relu", padding="same", name='Conv2DT_3')(x)
    x = layers.Conv2DTranspose(32, (16, 16), strides=2, activation="relu", padding="same", name='Conv2DT_4')(x)
    x = layers.Conv2D(1, (16, 16), activation="sigmoid", padding="same", name='Conv2D_Out')(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()

    autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=1,
    batch_size=1,
    shuffle=True,
    validation_data=(test_data, test_data)
    )

    predictions = autoencoder.predict(test_data)
    display1(test_data, predictions)

    
    layer = autoencoder.get_layer(name=layer_name)
    feature_extractor = Model(inputs=autoencoder.inputs, outputs=layer.output)

    # Compute image inputs that maximize per-filter activations
    # for the first 32 filters of our target layer
    all_imgs = []
    for filter_index in range(32):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index, feature_extractor)
        #print(img)

        w, h = 512, 512
        data = np.zeros((h, w, 1), dtype=np.uint8)
        data[0:513, 0:513] = img
        plt.imshow(data, interpolation='nearest')
        plt.savefig('fig' + str(len(all_imgs)) + '.png')
        plt.show()

        all_imgs.append(img)

    """
    # Build a black picture with enough space for
    # our 3 x 3 filters of size 128 x 128, with a 5px margin in between
    margin = 6
    n = 3
    cropped_width = img_width - 3 * 2
    cropped_height = img_height - 3 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    print(width, height)
    stitched_filters = np.zeros((width, height, 1))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)

    display(Image("stiched_filters.png"))
    """
    




main()