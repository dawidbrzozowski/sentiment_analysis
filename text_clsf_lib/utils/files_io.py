import json
import pickle
import numpy as np


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def write_json(out_path, data):
    with open(out_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def append_json(out_path, data):
    json_data: dict = load_json(out_path)
    assert type(json_data) is dict and type(data) is dict, "Can't append json data, if the data is not a dict."
    json_data.update(data)
    write_json(out_path, json_data)


def write_pickle(out_path, obj):
    pickle.dump(obj, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    return pickle.load(open(path, 'rb'))


def write_numpy(path, np_obj):
    np.save(path, np_obj)


def read_numpy(path):
    return np.load(path)
