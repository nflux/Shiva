import numpy as np

# def np_softmax(x):
#     """
#     Compute softmax values for each sets of scores in x.
#
#     Rows are scores for each class.
#     Columns are predictions (samples).
#     """
#     scoreMatExp = np.exp(np.asarray(x))
#     return scoreMatExp / scoreMatExp.sum(0)

def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)