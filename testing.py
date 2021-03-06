import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# The dimensions of our input image
img_width = 9
img_height = 9
n_filters = 2
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = 'Conv2DT_1'

def preprocess(array, isY):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    array = np.array(array)
    #array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), img_height, img_width, 1)) if isY else np.reshape(array, (1, img_height, img_width, n_filters))
    return array


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
        plt.imshow(image1.reshape(img_height, img_width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(img_height, img_width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

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
    # We run gradient ascent for 30 steps
    iterations = 25000
    learning_rate = 1.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    #print(img)
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


def load_image_data(image_path):
    """ 
    Load the raw image data into a data matrix and target vector.
    """
    x = []

    im = np.array(load_img(image_path))
    for i in range(len(im[0])):
        for j in range(len(im[0][i])):
            if im[0][i][j] != 255:
                im[0][i][j] = 0

    im = tf.image.resize(im, size=(img_height, img_width)).numpy()
    im = tf.image.rgb_to_grayscale(im)

    x.append(im)
    x = np.array(x)
    return x


def coalesce(x):
    #Using numpy will stop gradient tape computations, but do i need the gradients to be computed in the output layer?
    y = tf.math.reduce_sum(x, tf.rank(x)-1)
    return y


def main():

    # Since we only need images from the dataset to encode and decode, we
    # won't use the labels.
    
    """
    y = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 1, 1, 1, 0, 0], 
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    y = np.transpose(y)

    x1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    x2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    """

    y = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 1, 1, 1, 0], 
                   [0, 1, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    y = np.transpose(y)

    x1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    x2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 1, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    

    x_data = [x1, x2]

    y_data = [y]


    # Normalize and reshape the data
    x_data = preprocess(x_data, False)
    y_data = preprocess(y_data, True)
    

    # ADJUST THESE FOR DIFFERENT TESTS
    filter_x, filter_y = 3, 3
    output_path = f'recent_testing/line/{n_filters}_filters_/line/'


    if not os.path.exists(output_path):
        os.makedirs(output_path)


    input = layers.Input(shape=(img_height, img_width, n_filters))

    # Encoder
    #x = layers.Conv2D(n_filters, (filter_x, filter_y), strides=(1, 1), activation="linear", padding="same", use_bias=False, name='Conv2D_1')(input) # Removed kernal reg, maybe 

    # Decoder
    x = layers.Conv2DTranspose(1, (filter_x, filter_y), input_shape=(img_height, img_width, n_filters), strides=(1, 1), activation="linear", use_bias=False, padding="same", name='Conv2DT_1')(input)


    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss="mse")
    autoencoder.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)


    autoencoder.fit(
    x=x_data,
    y=y_data,
    epochs=10000,
    batch_size=1,
    shuffle=False,
    validation_data=(x_data, y_data),
    callbacks=[callback]
    )


    #plt.imshow(train_data[0], cmap='gray')
    #plt.show()
    #plt.clf()
    
    Ys = []
    """
    for i in range(n_filters):
        filters = autoencoder.get_layer(name=layer_name).get_weights()
        test_model = keras.models.clone_model(autoencoder)
        test_filters = filters[0]
        print(np.shape(test_filters))
        test_filters[:, :, :, i] = np.reshape([0.0] * (filter_x * filter_y), (filter_x, filter_y, 1))
        #test_biases = biases
        #test_biases[i] = 0.0
        test_model.get_layer(name=layer_name).set_weights([test_filters])

        img1 = autoencoder(test_data, training=False)
        img2 = test_model(test_data, training=False)

        Y = float(np.square(np.subtract(img1,img2)).mean())
        Ys.append(Y)
        print('MSE without filter ' + str(i) + ':', Y)

    plt.bar([x for x in range(len(Ys))], Ys)
    plt.savefig(output_path + 'filter_importance_' + str(filter_x) + '.png')
    plt.clf()



    filters = autoencoder.get_layer(name=layer_name).get_weights()
    # normalize filter values to 0-1 so we can visualize them
    filters = filters[0]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    ix = 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, 0], cmap='gray')
        print(f[:, :, 0])
        ix += 1
    # show the figure
    plt.savefig(output_path + 'filters_' + str(filter_x) + '.png')
    #plt.show()
    plt.clf()
    


    from numpy import expand_dims
    model = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer(name='Conv2D_1').output)
    # load the image with the required shape
    img = test_data[0]
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # get feature map for first hidden layer
    feature_maps = model(img, training=False)
    # plot the output from each block
    square = 1
    # plot all 8 outputs in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            if ix <= n_filters:
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                #im = ax.imshow(feature_maps[0, :, :, ix-1], interpolation='nearest')
                # plot filter channel in grayscale
                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="5%", pad=0.05)
                #plt.colorbar(im, cax=cax)
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray') #[0, :, :, ix-1]
                print(feature_maps[0, :, :, ix-1])
                ix += 1
    # show the figure
    plt.savefig(output_path + 'output_map_' + str(filter_x) + '.png')
    #plt.show()
    plt.clf()
    

    
    layer = autoencoder.get_layer(name=layer_name)
    feature_extractor = Model(inputs=autoencoder.inputs, outputs=layer.output)

    activation_path = output_path + 'activation_maps_' + str(filter_x) + '/'
    if not os.path.exists(activation_path):
        os.makedirs(activation_path)

    # Compute image inputs that maximize per-filter activations
    # for the first n_filter filters of our target layer
    all_imgs = []
    i = 0
    for filter_index in range(n_filters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index, feature_extractor)

        data = np.zeros((img_width, img_height, 1), dtype=np.uint8)
        data[0:img_width + 1, 0:img_height + 1] = img
        ax = plt.subplot()
        im = ax.imshow(data, interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(activation_path + str(i) + '.png')
        #plt.show()
        plt.clf()


        i += 1
        all_imgs.append(img)
    """

main()