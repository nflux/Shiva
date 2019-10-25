import pickle, json


def save_pickle_obj(obj, filename):
    '''
        Saves a Python object

        Input
            @obj        Instance to save
            @filename   Absolute path where to save including filename
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_obj(filename):
    '''
        Loads a Python object

        Input
            @filename   Absolute path to the file
    '''
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def save_to_json(data, filename):
    '''
        Saves data to JSON

        Input
            @data       Data to be serialized as a JSON
            @filename   Absolute path where to save including filename
    '''
    with open(filename, 'w') as handle:
        json.dump(data, handle)

def load_from_json(filename):
    '''
        Loads a JSON

        Input
            @filename   Absolute path to the file
    '''
    with open(filename, 'r') as handle:
        return json.load(handle)
