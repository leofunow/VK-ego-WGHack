from config import *
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error


def fit(df, max_depth = 5, random_state = 42, colnames = DF_ATTR):
    rgrs = RandomForestRegressor(max_depth = max_depth, random_state=random_state)
    rgrs.fit(df[colnames], df['x1'])
    return rgrs

def predict(df, rgrs, colnames = DF_ATTR):
    return rgrs.predict(df[colnames])

def export_p(filename, rgrs):
    with open(filename, 'wb') as f:
        pickle.dump(rgrs, f)

def import_p(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def rmse(df, pred):
    return mean_squared_error(df['x1'], pred, squared=False)