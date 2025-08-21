from tensorflow import keras
from functions import get_model, train_model, load_datasets

def main():
    model = get_model()

    dataset_name = "emnist/balanced"

    train_images, train_labels, test_images, test_labels = load_datasets(dataset_name)

    model = train_model(model, train_images, train_labels, test_images, test_labels)

    model_name = "emnist_model.keras"

    model.save(model_name)

    pass    

if __name__ == "__main__":
    main()