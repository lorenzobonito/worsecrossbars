"""simulation:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""
import gc
import logging
from typing import List
from typing import Tuple

import numpy as np
from numpy import ndarray
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import train_mlp
from worsecrossbars.backend.weights_manipulation import alter_weights
from worsecrossbars.backend.weights_manipulation import discretise_weights
from worsecrossbars.backend.weights_manipulation import join_weights
from worsecrossbars.backend.weights_manipulation import split_weights


def fault_simulation(
    percentages: ndarray,
    network_weights: List[ndarray],
    network_model: Model,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
) -> ndarray:
    """This function runs a fault simulation with the given parameters, and thus constitutes the
    computational core of the package. Each simulation is run "number_simulations" times, to average
    out stochastic variability in the final results.

    Args:
      percentages: Array containing the various percentages of faulty devices the user wants
        to simulate.
      network_weights: Array containing trained weights that are to be altered to simulate faults.
      network_model: Keras model containing the network topology being simulated.
      dataset: MNIST test dataset, used to calculate inference accuracy.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    Returns:
      accuracies: Array containing the ANN's inference accuracy at each percentage of faulty
      devices.
    """

    positive_cba, negative_cba = split_weights(network_weights)

    discretised_positive_cba = discretise_weights(positive_cba, simulation_parameters)
    discretised_negative_cba = discretise_weights(negative_cba, simulation_parameters)

    accuracies = np.zeros(len(percentages))

    for _ in range(simulation_parameters["number_simulations"]):

        accuracies_list = []

        for percentage in percentages:

            altered_positive_weights = alter_weights(
                discretised_positive_cba, percentage, simulation_parameters
            )
            altered_negative_weights = alter_weights(
                discretised_negative_cba, percentage, simulation_parameters
            )

            # The "set_weights" function sets the ANN's weights to the values specified in the
            # list of arrays "altered_weights"
            network_model.set_weights(
                join_weights(altered_positive_weights, altered_negative_weights)
            )
            accuracies_list.append(
                network_model.evaluate(dataset[1][0], dataset[1][1], verbose=0)[1]
            )

        accuracies += np.array(accuracies_list)
        gc.collect()

    accuracies /= simulation_parameters["number_simulations"]

    return accuracies


def train_models(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
    epochs: int,
    batch_size: int,
) -> Tuple[List[List[ndarray]], List[History]]:
    """This function trains the generated Keras models on the MNIST dataset with the given
    parameters.

    Args:
      dataset: MNIST dataset used to train the models.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.
      epochs: Positive integer, number of epochs over which the models will be trained.
      batch_size: Positive integer, size of batches that will be used to train the models.

    Returns:
      weights_list: List of arrays containing the weights of each of the "number_ANNs" trained
        networks.
      histories_list: List of Keras history dictionaries for each of the "number_ANNs" trained
        networks.
    """

    fault_type = simulation_parameters["fault_type"]
    number_anns = simulation_parameters["number_ANNs"]
    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    noise_variance = simulation_parameters["noise_variance"]

    weights_list = []
    histories_list = []

    for model_number in range(0, int(number_anns)):

        model = mnist_mlp(number_hidden_layers, noise_variance=noise_variance)
        mlp_weights, mlp_history, *_ = train_mlp(dataset, model, epochs, batch_size)
        weights_list.append(mlp_weights)
        histories_list.append(mlp_history)

        gc.collect()

        logging.info(
            "[%dHL_%s_%.2fNV] Trained model %d of %d.",
            number_hidden_layers,
            fault_type,
            noise_variance,
            model_number + 1,
            number_anns,
        )

    return weights_list, histories_list


def training_validation_metrics(
    histories_list: List[History],
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """This function calculates training and validation metrics by averaging the data generated
    during model training and stored in the Keras histories dictionaries returned by the
    train_models function.

    Args:
      histories_list: List containing Keras models' training histories, as output by the fit
        method run by the worsecrossbars.backend.mlp_trainer.train_mlp function.

    Returns:
      training_accuracy_values: Array containing the training accuracy values.
      validation_accuracy_values: Array containing the validation accuracy values.
      training_loss_values: Array containing the training loss values.
      validation_loss_values: Array containing the validation loss values.
    """

    training_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    training_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))

    for history in histories_list:

        history_dict = history.history
        training_accuracy_values += np.array(history_dict["accuracy"])
        validation_accuracy_values += np.array(history_dict["val_accuracy"])
        training_loss_values += np.array(history_dict["loss"])
        validation_loss_values += np.array(history_dict["val_loss"])

    training_accuracy_values /= len(histories_list)
    validation_accuracy_values /= len(histories_list)
    training_loss_values /= len(histories_list)
    validation_loss_values /= len(histories_list)

    return (
        training_accuracy_values,
        validation_accuracy_values,
        training_loss_values,
        validation_loss_values,
    )


def run_simulation(
    weights_list: List[List[ndarray]],
    percentages: ndarray,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
) -> ndarray:
    """This function runs the main simulation. This entails running one full fault_simulation for
    each of the "number_ANNs" networks trained above, so that the accuracies resulting from each can
    be averaged together to reduce the influence of stochastic variability.

    Args:
      weights_list: List of arrays containing the weights of each of the "number_ANNs" trained
        networks.
      percentages: Array containing the various percentages of faulty devices the user wants
        to simulate.
      dataset: MNIST test dataset, used to calculate inference accuracy.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    Returns:
      accuracies_array: Array containing the average of all accuracies obtained for each of the
        "number_ANNs" networks (each itself run "number_simulations" times), for a total of
        "number_ANNs" * "number_simulations" datapoints averaged together to obtain each value
        stored in the final array.
    """

    fault_type = simulation_parameters["fault_type"]
    number_anns = simulation_parameters["number_ANNs"]
    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    noise_variance = simulation_parameters["noise_variance"]

    model = mnist_mlp(number_hidden_layers, noise_variance=noise_variance)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    accuracies_array = np.zeros((len(weights_list), len(percentages)))

    for count, weights in enumerate(weights_list):

        accuracies_array[count] = fault_simulation(
            percentages, weights, model, dataset, simulation_parameters
        )

        logging.info(
            "[%dHL_%s_%.2fNV] Simulated model %d of %d.",
            number_hidden_layers,
            fault_type,
            noise_variance,
            count + 1,
            number_anns,
        )
        gc.collect()

    return np.mean(accuracies_array, axis=0, dtype=np.float64)
