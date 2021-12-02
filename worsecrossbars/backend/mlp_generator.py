"""mlp_generator:
A backend module used to create a Keras model for a densely connected MLP with a given topology.
"""
from typing import List

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential


def mnist_mlp(
    num_hidden_layers: int,
    neurons: List[int] = None,
    model_name: str = "",
    noise_variance: float = 0.0,
) -> Model:
    """This function returns a Keras model set up to be trained to recognise digits from the MNIST
    dataset (784 input neurons, 10 softmax output neurons).

    The network architecture corresponds to that employed in the "Simulation of Inference Accuracy
    Using Realistic RRAM Devices" paper. It consists of a feed-forward multilayer perceptron with
    784 input neurons (encoding pixel intensities for 28 × 28 pixel MNIST images), two 100-neuron
    hidden layers, and 10 output neurons (each corresponding to one of the ten digits). The first
    three layers employ a sigmoid activation function, whilst the output layer makes use of a
    softmax activation function (which means a cross-entropy error function is then used
    throughout the learning task). All 60,000 MNIST training images were employed, divided into
    training and validation sets in a 3:1 ratio, as described in the aforementioned paper.

    For the one-layer, three-layers and four-layers topologies, the number of neurons in each
    hidden layer was tweaked so as to produce a final network with about the same number of
    trainable parameters as the original, two-layers ANN. This was done to ensure that variability
    in fault simulation results was indeeed due to the number of layers being altered, rather than
    to a different number of weights being implemented.

    The function also gives the user the option to add GaussianNoise layers with a specific variance
    between hidden layers during training. This is done to increase the network's generalisation
    power, as well as to increase resilience to faulty memristive devices.

    Args:
      num_hidden_layers: Integer comprised between 1 and 4, number of hidden layers instantiated as
        part of the model.
      neurons: List of length num_hidden_layers, contains the number of neurons to be created in
        each densely-connected layer.
      model_name: String, name of the Keras model.
      noise_variance: Positive integer/float, variance of the GaussianNoise layers instantiated
        during training to boost network performance.

    Returns:
      model: Keras model object, contaning the desired topology.
    """

    default_neurons = {1: [112], 2: [100, 100], 3: [90, 95, 95], 4: [85, 85, 85, 85]}

    # Setting default argument values
    if neurons is None:
        neurons = default_neurons[num_hidden_layers]
    if model_name == "":
        model_name = f"MNIST_MLP_{num_hidden_layers}HL"

    if not isinstance(neurons, list) or len(neurons) != num_hidden_layers:
        raise ValueError(
            '"neurons" argument should be a list object with the same length as '
            + "the number of layers being instantiated."
        )

    if not isinstance(model_name, str):
        raise ValueError('"model_name" argument should be a string object.')

    model = Sequential(name=model_name)

    # Creating first hidden layer
    model.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
    if noise_variance:
        model.add(GaussianNoise(noise_variance))
    model.add(Activation("sigmoid"))

    # Creating other hidden layers
    for layer_index, neuron in enumerate(neurons[1:]):

        model.add(Dense(neuron, name=f"{model_name}_L{layer_index+2}"))
        if noise_variance:
            model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))

    # Creating output layer
    model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return model
