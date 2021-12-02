"""dp_testing:
A testing module used to benchmark differential pairs functionality.
"""
import sys
from typing import List

import numpy as np
from numpy import ndarray

import worsecrossbars.backend.weights_manipulation as wm
from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import create_datasets
from worsecrossbars.backend.mlp_trainer import train_mlp


def test_split_weights(network_weights: List[ndarray]) -> None:
    """This function can be used to test whether the split_weights function contained in the
    weights_manipulation module is working as intended.

    Args:
      network_weights: The weights as output by the training functions.
    """

    test = True
    crossbar_one, crossbar_two = wm.split_weights(network_weights)

    for count, layer in enumerate(network_weights):

        if count % 2 == 0:
            for idx1, lst in enumerate(crossbar_one[count]):
                for idx2, item in enumerate(lst):
                    test &= item in (layer[idx1][idx2], 0.0)
            for idx1, lst in enumerate(crossbar_two[count]):
                for idx2, item in enumerate(lst):
                    test &= item in (-layer[idx1][idx2], 0.0)
        else:
            test &= np.array_equal(layer, crossbar_one[count]) and np.array_equal(
                layer, crossbar_two[count]
            )

    if not test:
        print("split_weights functions failed. Accessing trace ...")
        sys.exit(1)


def test_join_weights(
    original_weights: List[ndarray], crossbar_one: List[ndarray], crossbar_two: List[ndarray]
) -> None:
    """This function can be used to test whether the join_weights function contained in the
    weights_manipulation module is working as intended.

    Args:
      original_weights: The original weights array.
      crossbar_one: List containing network weights to be programmed onto the crossbar array with
        positive input voltages.
      crossbar_two: List containing network weights to be programmed onto the crossbar array with
        negative input voltages.
    """

    test = True
    joined_weights = wm.join_weights(crossbar_one, crossbar_two)

    for original_layer, joined_layer in zip(original_weights, joined_weights):
        test &= np.array_equal(original_layer, joined_layer)

    if not test:
        print("join_weights functions failed. Accessing trace ...")
        sys.exit(1)


if __name__ == "__main__":

    simulation_parameters = {
        "HRS_LRS_ratio": 5,
        "number_conductance_levels": 10,
        "excluded_weights_proportion": 0.015,
        "number_hidden_layers": 1,
        "fault_type": "STUCKZERO",
        "noise_variance": 0,
        "number_ANNs": 10,
        "number_simulations": 10,
    }

    model = mnist_mlp(1, [5])
    dataset = create_datasets(training_validation_ratio=3)

    trained_weights, *_ = train_mlp(dataset, model, 10, 100)

    # Testing split_weights function
    test_split_weights(trained_weights)
    positive_cba, negative_cba = wm.split_weights(trained_weights)

    # Testing join_weights function
    test_join_weights(trained_weights, positive_cba, negative_cba)

    discretised_positive_cba = wm.discretise_weights(positive_cba, simulation_parameters)
    altered_weights = wm.alter_weights(discretised_positive_cba, 0.25, simulation_parameters)

    print(discretised_positive_cba)
    print(altered_weights)
