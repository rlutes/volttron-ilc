# -*- coding: utf-8 -*- {{{
# ===----------------------------------------------------------------------===
#
#                 Installable Component of Eclipse VOLTTRON
#
# ===----------------------------------------------------------------------===
#
# Copyright 2022 Battelle Memorial Institute
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# ===----------------------------------------------------------------------===
# }}}

import logging
import numpy as np

from collections import OrderedDict
from scipy.optimize import linprog

from collections import defaultdict

from importlib.metadata import distribution, PackageNotFoundError
try:
    distribution('volttron-core')
    from volttron.client.logs import setup_logging
except PackageNotFoundError:
    from volttron.platform.agent.utils import setup_logging

setup_logging()
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s   %(levelname)-8s %(message)s',
                    datefmt='%m-%d-%y %H:%M:%S')


def fucom(pairwise_comparisons: dict[str, dict[str, float]], out_debug_dict=None) -> dict[str: float]:
    """
    Takes a dictionary of ranked criteria and returns a dictionary with the optimal weights of these same criteria.
        Criteria should be ranked on a scale from 1 to 10, relative to the most significant criterion.
        Lower numbers represent greater significance than higher numbers.
        The most significant criterion should be ranked 1.

        An optional argument, out_debug_dict can be used as an output parameter to inspect the input used to linprog
          within the function. Pass an empty dictionary to out_debug_dict, and it will contain the parameters passed
          to linprog() once these are available (before linprog has run).
    """
    criteria_weights = {}
    for state, ranked_criteria in pairwise_comparisons.items():
        sorted_criteria = OrderedDict(sorted(ranked_criteria.items(), key=lambda item: item[1]))
        criteria_count = len(sorted_criteria)
        objective = [1] + [0]*criteria_count
        lhs_ineq = np.zeros([2*criteria_count-3, criteria_count+1])
        lhs_ineq[:,0] = -1  # First column represents the minimized variable.
        criteria_values = list(sorted_criteria.values())
        for c in range(criteria_count-1):
            lhs_ineq[c, c+1] = 1
            lhs_ineq[c, c+2] = -criteria_values[c+1] / criteria_values[c]
        for c in range(criteria_count-2):
            lhs_ineq[c+criteria_count-1, c+1] = 1
            lhs_ineq[c+criteria_count-1, c+3] = -criteria_values[c+2]/criteria_values[c]
        rhs_ineq = np.zeros([1, lhs_ineq.shape[0]])
        lhs_eq = [[0] + [1]*criteria_count]
        rhs_eq = [1]
        bounds = [(0, float('inf'))] + [(0, 1)]*criteria_count
        if isinstance(out_debug_dict, dict):
            # Debug parameters is not returned, but is available to the caller if a dict was passed in.
            out_debug_dict.update({"objective": objective, "lhs_ineq": lhs_ineq, "rhs_ineq": rhs_ineq,
                                   "lhs_eq": lhs_eq, "rhs_eq": rhs_eq, "bounds": bounds})
        result = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq,
                         bounds=bounds, method='revised simplex')
        weights = {k: float(result.x[i+1]) for i, k in enumerate(sorted_criteria.keys())}
        criteria_weights[state] = weights
    return criteria_weights


def extract_criteria(pairwise_configuration: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Extract pairwise criteria parameters
    :param pairwise_configuration:
    :return:
    """
    # check if file has been updated or uses old format
    _log.debug(f"Pairwise input data: {pairwise_configuration}")
    if "curtail" not in pairwise_configuration.keys() and "augment" not in pairwise_configuration.keys():
        config_matrix = {"curtail": pairwise_configuration}
    else:
        config_matrix = pairwise_configuration
    return config_matrix


def build_score(_matrix, weight, priority):
    """
    Calculates the curtailment score using the normalized matrix
    and the weights vector. Returns a sorted vector of weights for each
    device that is a candidate for curtailment.
    :param _matrix:
    :param weight:
    :param priority:
    :return:
    """
    input_keys, input_values = _matrix.keys(), _matrix.values()
    scores = []

    for input_array in input_values:
        criteria_sum = sum(i*w for i, w in zip(input_array, weight))

        scores.append(criteria_sum*priority)

    return zip(scores, input_keys)


def _verify_criteria_match(criteria_labels: list[str], device_criteria: list[str]) -> None:
    """
    Verifies that the input criteria labels match the data criteria.
    """
    if set(device_criteria) != set(criteria_labels):
        raise Exception("Input criteria and data criteria do not match")


def _normalize_values(builder: dict[str, dict[str, float]], sums_by_criteria: dict[str, float],
                      criteria_labels: list[str]) -> dict[str, list[float]]:
    """
    Normalizes the input data values based on the sum of criteria.
    """
    normalized_matrix = {}
    for key, device_data in builder.items():
        normalized_matrix[key] = [
            (device_data[tag] / sums_by_criteria[tag]) if device_data[tag] else 0.0
            for tag in criteria_labels
        ]
    return normalized_matrix


def input_matrix(device_data_builder: dict[str, dict[str, float]], criteria_labels: list[str]) -> dict[
    str, list[float]]:
    """
    Constructs and returns a normalized input matrix from device data.

    :param device_data_builder: Dictionary containing device data.
    :param criteria_labels: List of criteria labels to use for the matrix.
    :return: A dictionary representing the normalized input matrix.
    """
    # Validate criteria match
    device_criteria = list(next(iter(device_data_builder.values())).keys())
    _verify_criteria_match(criteria_labels, device_criteria)

    # Compute sums for each criterion
    sums_by_criteria = defaultdict(float)
    for device_data in device_data_builder.values():
        for criterion, value in device_data.items():
            sums_by_criteria[criterion] += value

    # Normalize values
    return _normalize_values(device_data_builder, sums_by_criteria, criteria_labels)
