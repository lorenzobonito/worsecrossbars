"""weights_manipulation:
A backend module used to map ANN weights into real-world resistance levels.
"""
import copy
from typing import List
from typing import Tuple

import numpy as np
from numpy import ndarray


def choose_extremes(
    network_weights: List[ndarray], hrs_lrs_ratio: float, excluded_weights_proportion: float
) -> List[Tuple[float, float]]:
    """This function chooses the minimum and maximum discrete weights of the memristor ANN.

    Args:
      network_weights: The weights as outputted by the training functions.
      hrs_lrs_ratio: The desired HRS/LRS ratio.
      excluded_weights_proportion: The proportion of weights at the tails to be excluded.

    Returns:
      extremes_list: The list of extremes at each layer of the network.
    """

    extremes_list = []

    for count, layer_weights in enumerate(network_weights):

        if count % 2 == 0:
            # Flattening and sorting array descendingly
            layer_weights = -np.sort(-layer_weights, axis=None)
            index = int(excluded_weights_proportion * layer_weights.size)
            w_max = layer_weights[index]
            w_min = w_max / hrs_lrs_ratio
            extremes_list.append((w_max, w_min))
        else:
            extremes_list.append((np.nan, np.nan))

    return extremes_list


def create_weight_interval(
    extremes_list: List[Tuple[float, float]],
    number_of_levels: int,
) -> List[ndarray]:
    """This function creates an evenly spaced weight interval.

    Args:
      extremes_list: The list of extremes in each layer.
      number_of_levels: The number of weights needed.

    Returns:
      weight_interval_list: A list of linearly spaced weight intervals for the synaptic layers.
    """

    weight_interval_list = []

    for count, element in enumerate(extremes_list):
        if count % 2 == 0:
            weight_interval_list.append(np.linspace(element[1], element[0], number_of_levels))
        else:
            weight_interval_list.append(np.empty(number_of_levels))

    return weight_interval_list


def discretise_weights(
    network_weights: List[ndarray], simulation_parameters: dict
) -> List[ndarray]:
    """This function alters the weights in the network so that they conform to the list of allowed
    weights.

    Args:
      network_weights: The weights as output by the training functions.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    Returns:
      discretised_weights: The altered network weights, now discretised.
    """

    discretised_weights = copy.deepcopy(network_weights)

    extremes_list = choose_extremes(
        network_weights,
        simulation_parameters["HRS_LRS_ratio"],
        simulation_parameters["excluded_weights_proportion"],
    )

    network_weight_intervals = create_weight_interval(
        extremes_list, simulation_parameters["number_conductance_levels"]
    )

    for count, layer_weights in enumerate(discretised_weights):

        if count % 2 == 0:

            # Checking which weights are equal to 0, and thus correspond to devices which shall not
            # be electroformed and, consequently, shall not undergo discretisation.
            zeros_mask = layer_weights != 0.0

            original_shape = layer_weights.shape
            layer_weights = layer_weights.flatten()
            req_int = network_weight_intervals[count]
            # req_int = np.concatenate((np.negative(req_int)[::-1], req_int), axis=None)
            index = np.searchsorted(req_int, layer_weights)
            mask = index > len(req_int) - 1
            index[mask] = len(req_int) - 1
            index_new = np.array(
                [
                    index[i] - 1
                    if abs(req_int[index[i] - 1] - layer_weights[i])
                    < abs(req_int[index[i]] - layer_weights[i])
                    else index[i]
                    for i in range(len(index))
                ]
            )
            layer_weights = np.array([req_int[_] for _ in index_new])
            layer_weights = np.reshape(layer_weights, original_shape)
            layer_weights *= zeros_mask
            discretised_weights[count] = layer_weights

    return discretised_weights


def alter_weights(
    network_weights: List[ndarray], failure_percentage: float, simulation_parameters: dict
) -> List[ndarray]:
    """This function takes in and modifies network weights to simulate the effect of faulty
    memristive devices being used in the physical implementation of the ANN.

    It should be noted that only synapse parameters (i.e. weights, not neuron biases) are being
    altered. This is achieved by only modifying even-numbered layers, given that, in a
    densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.

    Args:
      network_weights: Array containing the weights as outputted by the training functions.
      failure_percentage: Positive integer/float, percentage of devices affected by the fault.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    Returns:
      altered_weights: Array containing the ANN weights altered to simulate the effect of the
        desired percentage of devices being affected by the specified fault.
    """

    if isinstance(failure_percentage, int):
        failure_percentage = float(failure_percentage)

    if not isinstance(failure_percentage, float) or failure_percentage < 0:
        raise ValueError('"failure_percentage" argument should be a positive real number.')

    altered_weights = copy.deepcopy(network_weights)
    extremes_list = choose_extremes(
        network_weights,
        simulation_parameters["HRS_LRS_ratio"],
        simulation_parameters["excluded_weights_proportion"],
    )

    for count, layer_weights in enumerate(altered_weights):

        if count % 2 == 0:
            if simulation_parameters["fault_type"] == "STUCKZERO":
                fault_value = 0.0
            elif simulation_parameters["fault_type"] == "STUCKHRS":
                fault_value = extremes_list[count][0]
            else:
                fault_value = extremes_list[count][1]

            # Checking which weights are equal to 0, and thus correspond to devices which shall not
            # be electroformed and, consequently, shall not be altered.
            choice_probabilites = layer_weights.flatten() / layer_weights.sum()

            indices = np.random.choice(
                layer_weights.shape[0] * layer_weights.shape[1],
                replace=False,
                size=int(np.count_nonzero(layer_weights) * failure_percentage),
                p=choice_probabilites,
            )

            layer_weights[np.unravel_index(indices, layer_weights.shape)] = fault_value

    return altered_weights


def split_weights(network_weights: List[ndarray]) -> Tuple[List[ndarray], List[ndarray]]:
    """This function takes in the list of network weights obtained by training the model in
    software, and splits said weights into two lists, each to be implemented on a separate memristor
    crossbar array.

    Args:
      network_weights: The weights as output by the training functions.

    Returns:
      split_weights: Tuple containing two lists, each with the weights to be implemented on one of
        the two crossbar arrays. One CBA will be input a negative current, and be used to implement
        negative weights, and one will be input a positive current, and be used to implement
        positive weights. All weights in split_weights shall thus be positive.
    """

    # For a first attempt, the weights shall be split on the two crossbar arrays based on whether
    # they are positive or negative in value. Odd-numbered layers contain neuron biases and can
    # thus be ignored.

    positive_cba = []
    negative_cba = []

    for count, layer_weights in enumerate(network_weights):

        if count % 2 == 0:
            positive_cba.append(layer_weights.clip(min=0.0))
            negative_cba.append(np.abs(layer_weights.clip(max=0.0)))
        else:
            positive_cba.append(layer_weights)
            negative_cba.append(layer_weights)

    return positive_cba, negative_cba


def join_weights(positive_cba: List[ndarray], negative_cba: List[ndarray]) -> List[ndarray]:
    """This function takes in the weights of each of the two crossbar arrays and joins them into a
    single array of weights, so that the performance of a neural network implemented with said CBAs
    can be simulated.

    Args:
      positive_cba: List containing network weights to be programmed onto the crossbar array with
        positive input voltages.
      negative_cba: List containing network weights to be programmed onto the crossbar array with
        negative input voltages.

    Returns:
      joined_weights: List containing joined network weights for simulation purposes.
    """

    # For a first attempt, the weights shall simply be joined back together by subtraction

    joined_weights = []

    for count, layer_weights in enumerate(zip(positive_cba, negative_cba)):

        if count % 2 == 0:
            joined_weights.append(layer_weights[0] - layer_weights[1])
        else:
            joined_weights.append(layer_weights[0])

    return joined_weights
