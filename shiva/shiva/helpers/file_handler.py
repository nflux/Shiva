import pickle, json


def save_pickle_obj(obj, filename):
    """
    Save object as a pickle. Soon to be deprecated.

    Args:
        obj (object): instance to be saved
        filename (str): Absolute path where to save including filename

    Returns:
        None
    """
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_obj(filename):
    """
    Loads a pickle object. Soon to be deprecated.

    Args:
        filename (str): Absolute path to the file

    Returns:
        None
    """
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def save_to_json(data, filename):
    """
    Saves data to JSON

    Args:
        data (object): data to be serlaized as a JSON
        filename (str): Absolute path where to save including filename

    Returns:
        None
    """
    with open(filename, 'w') as handle:
        json.dump(data, handle)


def load_from_json(filename):
    """
    Loads a JSON

    Args:
        filename (str): Absolute path to the file

    Returns:
        None
    """
    with open(filename, 'r') as handle:
        return json.load(handle)
