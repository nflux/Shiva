import numpy as np

def np_softmax(x: np.ndarray, axis: int=0) -> np.ndarray:
    """
    Numpy approach to compute softmax values for each sets of scores in x

    Args:
        x (np.ndarray): numpy array to be softmaxed
        axis (int): axis where we are applying the softmax

    Returns:
        np.ndarray: new numpy array with the softmax values
    """
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def two_point_formula(x: float, p_1: set, p_2: set) -> float:
    x_1, y_1 = p_1
    x_2, y_2 = p_2
    m = (y_2 - y_1) / (x_2 - x_1)
    return m * (x - x_1) + y_1