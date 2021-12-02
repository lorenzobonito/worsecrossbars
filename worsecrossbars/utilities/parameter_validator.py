"""parameter_validator:
A utilities module used to assess the validity of the simulation parameters loaded from JSON dict.
"""


def validate_parameters(simulation_parameters: dict):
    """This function validates the dictionary containing the simulation parameters, by checking
    whether the values provided are of the correct type, and take on legal values.

    Args:
      simulation_parameters: Dictionary containing simulation parameters to be validated.
    """

    if not isinstance(simulation_parameters, dict):
        raise ValueError('"simulation_parameters" argument should be a dictionary object.')

    if simulation_parameters["fault_type"] not in ["STUCKZERO", "STUCKHRS", "STUCKLRS"]:
        raise ValueError(
            '"fault_type" argument should be valid string. Acceptable values include "STUCKZERO"'
            + '"STUCKHRS" and "STUCKLRS".'
        )

    if simulation_parameters["number_hidden_layers"] not in [1, 2, 3, 4]:
        raise ValueError('"number_hidden_layers" argument should be an integer between 1 and 4.')

    if (
        not isinstance(simulation_parameters["number_ANNs"], int)
        or simulation_parameters["number_ANNs"] < 1
    ):
        raise ValueError('"number_ANNs" argument should be an integer greater than or equal to 1.')

    if (
        not isinstance(simulation_parameters["number_simulations"], int)
        or simulation_parameters["number_simulations"] < 1
    ):
        raise ValueError(
            '"number_simulations" argument should be an integer greater than or equal to 1.'
        )

    if (
        not isinstance(simulation_parameters["number_conductance_levels"], int)
        or simulation_parameters["number_conductance_levels"] < 1
    ):
        raise ValueError(
            '"number_conductance_levels" argument should be an integer greater than or equal to 1.'
        )

    if isinstance(simulation_parameters["noise_variance"], int):
        simulation_parameters["noise_variance"] = float(simulation_parameters["noise_variance"])
    if (
        not isinstance(simulation_parameters["noise_variance"], float)
        or simulation_parameters["noise_variance"] < 0
    ):
        raise ValueError('"noise_variance" argument should be a positive real number.')

    if isinstance(simulation_parameters["HRS_LRS_ratio"], int):
        simulation_parameters["HRS_LRS_ratio"] = float(simulation_parameters["HRS_LRS_ratio"])
    if (
        not isinstance(simulation_parameters["HRS_LRS_ratio"], float)
        or simulation_parameters["HRS_LRS_ratio"] < 0
    ):
        raise ValueError('"hrs_lrs_ratio" argument should be a positive real number.')

    if isinstance(simulation_parameters["excluded_weights_proportion"], int):
        simulation_parameters["excluded_weights_proportion"] = float(
            simulation_parameters["excluded_weights_proportion"]
        )
    if (
        not isinstance(simulation_parameters["excluded_weights_proportion"], float)
        or simulation_parameters["excluded_weights_proportion"] < 0
        or simulation_parameters["excluded_weights_proportion"] > 1
    ):
        raise ValueError(
            '"excluded_weights_proportion" argument should be a real number between 0 and 1.'
        )
