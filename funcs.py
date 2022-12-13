import numpy
import math
from scipy.special import log_softmax


def cost_function(logits: "numpy.ndarray", target: "int") -> "float":

    return log_softmax(logits)[target]


def log_density(
    weights: "numpy.ndarray",
    data: "numpy.ndarray",
    targets: "numpy.ndarray",
) -> "float":

    data_size: "int" = targets.shape[0]

    scores: "float" = 0.0

    for index in range(data_size):

        logits: "numpy.ndarray" = weights.dot(data[index])

        scores += cost_function(logits=logits, target=targets[index])

    return scores


def acceptance_prob(
    previous_weights: "numpy.ndarray",
    new_weights: "numpy.ndarray",
    precision: "float",
    data: "numpy.ndarray",
    targets: "numpy.ndarray",
) -> "float":

    new_log_density: "float" = log_density(new_weights, data, targets)

    previous_log_density: "float" = log_density(previous_weights, data, targets)

    full_log_density: "float" = new_log_density - previous_log_density

    edit_weights: "numpy.ndarray" = (new_weights - previous_weights) * (
        new_weights + previous_weights
    )

    scores += -(precision / 2) * edit_weights.sum()

    return max(1, math.exp(full_log_density))
