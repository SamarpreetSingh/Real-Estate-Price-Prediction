import json
import pickle
import numpy as np

__locations = None
__model = None
__data_columns = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        ind = __data_columns.index(location.lower())
    except:
        ind = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath

    if ind >= 0:
        x[ind] = 1


    return float(__model.predict([x]))


def get_location_names():
    return __locations


def load_saved_artifacts():
    print('Loading artifacts ...')
    global __data_columns
    global __locations
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']

    __locations = __data_columns[3:]

    with open('./artifacts/real_estate.pickle', 'rb') as f:
        __model = pickle.load(f)

    print('Loading artifacts done !')


if __name__ == '__main__':
    load_saved_artifacts()
    get_location_names()

    print(get_estimated_price('1st phase JP nagar', 1000, 3, 3))
