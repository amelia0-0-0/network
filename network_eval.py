import tensorflow_datasets as tfds
from tensorflow import keras
from functions import separate_labels_images

def main():
    datasets = tfds.load("emnist/mnist")["test"]

    model = keras.models.load_model("emnist_model.keras")

    test_images, test_labels = separate_labels_images(datasets)

    results = model.evaluate(test_images, test_labels, batch_size=128)

    print("test loss, test acc:", results)

    pass

if __name__ == "__main__":
    main()