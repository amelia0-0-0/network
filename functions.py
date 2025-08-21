import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

def separate_labels_images(dataset):
    """
    Separates images and labels from a TensorFlow dataset.

    Args:
        dataset: A TensorFlow dataset containing images and labels.

    Returns:
        Tuple of numpy arrays (images, labels).
    """
    images = []
    labels = []
    for example in tfds.as_numpy(dataset):
        images.append(example["image"])
        labels.append(example["label"])
    images = np.array(images)
    labels = np.array(labels)

    images = np.squeeze(images) / 255.0

    return images, labels


def get_model():
    """    Creates a simple Keras model for image classification.

    Returns:
        A Keras Sequential model.
    """

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),     # input layer (1)
        keras.layers.Dense(128, activation='relu'),     # hidden layer (2)
        keras.layers.Dense(128, activation='relu'),     # hidden layer (3)
        keras.layers.Dense(128, activation='relu'),     # hidden layer (4)
        keras.layers.Dense(47, activation='softmax')    # output layer (5)
    ])

    return model

def train_model(model, train_images, train_labels, test_images, test_labels):
    """    Trains the model with the provided training and testing datasets.

    Args:
        model: The Keras model to be trained.
        train_images: Training images.
        train_labels: Training labels.
        test_images: Testing images.
        test_labels: Testing labels.

    Returns:
        The trained model.
    """
    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

    return model

def load_datasets(dataset_name="emnist/balanced"):
    """    Loads the specified dataset and separates images and labels.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Tuple of numpy arrays (train_images, train_labels, test_images, test_labels).
    """

    datasets = tfds.load(dataset_name)

    train_dataset, test_dataset = datasets["train"], datasets["test"]

    train_images, train_labels = separate_labels_images(train_dataset)

    test_images, test_labels = separate_labels_images(test_dataset)

    return train_images, train_labels, test_images, test_labels