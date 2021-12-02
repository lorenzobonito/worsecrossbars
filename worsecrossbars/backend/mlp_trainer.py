"""mlp_trainer:
A backend module used to create the MNIST dataset and train a Keras model on it.
"""
from typing import List
from typing import Tuple

from numpy import ndarray
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def create_datasets(
    training_validation_ratio: float,
) -> Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """This function creates traning and validation datasets based on the MNIST digit database,
    according to the given training/validation split.

    Args:
      training_validation_ratio: Positive integer/float, ratio between size of training and
        validation datasets, indicating that the training dataset is "training_validation_ratio"
        times bigger than the validation dataset.

    Returns:
      validation_data: Array containing validation data.
      validation_labels: Array containing validation labels.
      training_data: Array containing training data.
      training_labels: Array containing training labels.
      test_data: Array containing test data.
      test_labels: Array containing test labels.
    """

    if isinstance(training_validation_ratio, int):
        training_validation_ratio = float(training_validation_ratio)

    if not isinstance(training_validation_ratio, float) or training_validation_ratio < 0:
        raise ValueError('"training_validation_ratio" argument should be a positive real number.')

    # Dataset download
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Data reshaping
    train_data = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_data = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Creating a validation set
    validation_size = round(train_data.shape[0] / (training_validation_ratio + 1))
    validation_data = train_data[:validation_size]
    training_data = train_data[validation_size:]
    validation_labels = train_labels[:validation_size]
    training_labels = train_labels[validation_size:]

    return (validation_data, validation_labels, training_data, training_labels), (
        test_data,
        test_labels,
    )


def train_mlp(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    model: Model,
    epochs: int,
    batch_size: int,
) -> Tuple[List[ndarray], History, float, float]:
    """This function trains a given Keras model on the dataset provided to it.

    Args:
      dataset: Tuple of tuples, containing training, validation and testing images and labels, as
        provided by the create_datasets() function defined above.
      model: Keras model which is to be trained
      epochs: Positive integer, number of epochs used in training.
      batch_size: Positive integer, number of batches used in training.

    Returns:
      mlp_weights: List containing the parameters associated with each layer of the ANN.
        mlp_weights[0] comprises the weights related to the synapses between layer 1 and layer 2,
        mlp_weights[1] contains the bias terms for the neurons in layer 2, and so forth.

      mlp_history: Keras history object, containing information regarding network performance at
        different epochs.

      mlp_test_loss: Final test loss

      mlp_test_acc: Final test accuracy
    """

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('"epochs" argument should be an integer greater than 1.')

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('"batch_size" argument should be an integer greater than 1.')

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # Training with validation test
    mlp_history = model.fit(
        dataset[0][2],
        dataset[0][3],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(dataset[0][0], dataset[0][1]),
        verbose=0,
    )
    mlp_test_loss, mlp_test_acc = model.evaluate(dataset[1][0], dataset[1][1], verbose=0)

    # Extracting network weights
    mlp_weights = model.get_weights()

    return mlp_weights, mlp_history, mlp_test_loss, mlp_test_acc
