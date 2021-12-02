"""json_handlers:
An internal utility module to handle and validate JSON.
"""
import logging
import sys

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def validate_json(extracted_json: dict) -> None:
    """Validate a given JSON object against a standard schema.

    Args:
      extracted_json: The JSON object to be validated.
    """

    json_schema = {
        "type": "object",
        "properties": {
            "simulations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "HRS_LRS_ratio": {"type": "integer"},
                        "number_conductance_levels": {"type": "integer"},
                        "excluded_weights_proportion": {"type": "number"},
                        "number_hidden_layers": {"type": "integer"},
                        "fault_type": {"type": "string"},
                        "noise_variance": {"type": "number"},
                        "number_ANNs": {"type": "integer"},
                        "number_simulations": {"type": "integer"},
                    },
                },
            },
            "accuracy_plots_parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plots_data": {"type": "array", "items": {"type": "string"}},
                        "xlabel": {"type": "string"},
                        "title": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                },
            },
            "training_validation_plots_parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plots_data": {"type": "array", "items": {"type": "string"}},
                        "title": {"type": "string"},
                        "filename": {"type": "string"},
                        "value_type": {"type": "string"},
                    },
                },
            },
        },
    }

    try:
        validate(extracted_json, json_schema)
    except ValidationError as err:
        logging.exception("An exception occurred while validating the JSON file.")
        print(f"{err.message}")
        sys.exit(0)
