import numpy
from scipy.special import log_softmax


def cost_function(logits: "numpy.ndarray", target: "int") -> "float":

    return log_softmax(logits)[target]


def log_density(
    weights: "numpy.ndarray",
    precision: "float",
    data: "numpy.ndarray",
    targets: "numpy.ndarray",
) -> "float":

    data_size: "int" = targets.shape[0]

    scores: "float" = 0.0

    for index in range(data_size):

        logits: "numpy.ndarray" = weights.dot(data[index])

        scores += cost_function(logits=logits, target=targets[index])

    scores += -(precision / 2) * (weights**2).sum()

    return scores
