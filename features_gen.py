import pandas as pd
import numpy as np


# Data path
data_path = './data/Full_data_Trentino.csv'

taxa = [
    'Ambrosia', 'Artemisia', 'Betula', 'Corylus', 'Cupressaceae, Taxaceae',
    'Fraxinus', 'Olea europaea', 'Ostrya carpinifolia', 'Poaceae', 'Urticaceae'
]

def get_features(taxa: list[str]=taxa):

    # Loading data
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])

    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.dayofyear

    # Taxa concentrations

    # Meteorological features
    meteo_features = ['temp_max', 'temp_min', 'temp_mean', 'rain', 'humidity', 'wind_dir', 'wind_speed', 'wind_gusts', 'rad', 'sun_hours', 'pressure']

    # Defining new input Features

    # Defining the different time windows (1/2 weeks, 1/3/6 months)
    time_windows = {
        '1w': 7,
        '2w': 14,
        '1m': 30,
        '3m': 90,
        '6m': 180
    }


    # Replace all '--' occurrences with previous day values
    data.replace('--', np.nan, inplace=True)
    data.replace('', np.nan, inplace=True)
    data.replace(' ', np.nan, inplace=True)
    data.ffill(inplace=True)

    # change all columns to float except datetime
    for column in data.columns:
        if column not in ['datetime']:
            data[column] = data[column].astype('float32')



    ### Add features here

    features_data = {}

    for f in meteo_features + taxa:
        for window_name, window_size in time_windows.items():
            features_data[f'{f}_rolling_mean_{window_name}'] = data[f].rolling(window=window_size, min_periods=1).mean()
            features_data[f'{f}_rolling_var_{window_name}'] = data[f].rolling(window=window_size, min_periods=1).var()



    for f in meteo_features + taxa:
        for window_name, window_size in time_windows.items():
            features_data[f'{f}_rolling_mean_{window_name}_delta'] = features_data[f'{f}_rolling_mean_{window_name}'] - features_data[f'{f}_rolling_mean_{window_name}'].shift(window_size)
            features_data[f'{f}_rolling_var_{window_name}_delta'] = features_data[f'{f}_rolling_var_{window_name}'] - features_data[f'{f}_rolling_var_{window_name}'].shift(window_size)

            if f in taxa:
                for i in range(2, 6):
                    features_data[f'{f}_rolling_mean_{window_name}_delta_{i}w'] = features_data[f'{f}_rolling_mean_{window_name}'] - features_data[f'{f}_rolling_mean_{window_name}'].shift(i)
                    features_data[f'{f}_rolling_var_{window_name}_delta_{i}w'] = features_data[f'{f}_rolling_var_{window_name}'] - features_data[f'{f}_rolling_var_{window_name}'].shift(i)
            else:
                # Adding this type of meteo features lowers the performance of the model
                # TODO: find new meteo features to add
                pass


    for t in taxa:
        kernel = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
        features_data[f'{t}_convolution'] = np.convolve(data[t], kernel, mode="same")


    final_features_df = pd.DataFrame(features_data)
    data = pd.concat([data, final_features_df], axis=1)
    data.dropna(inplace=True)

    features = [f for f in data.keys() if f not in ['datetime', 'year', 'month', 'day']]
    return data, features


