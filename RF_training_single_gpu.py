#Libraries for data processing
import numpy as np
import pandas as pd

#Library for plotting
import matplotlib.pyplot as plt

#Libraries for model training and validation
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import cuml.ensemble.randomforestregressor as cuml_rf
import xgboost as xgb

#Data path
data_path = './data/Full_data_Trentino.csv'

#Loading data
data = pd.read_csv(data_path)
data['datetime'] = pd.to_datetime(data['datetime'])

data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.dayofyear

#Taxa concentrations
taxa = [
    'Ambrosia', 'Artemisia', 'Betula', 'Corylus', 'Cupressaceae, Taxaceae',
    'Fraxinus', 'Olea europaea', 'Ostrya carpinifolia', 'Poaceae', 'Urticaceae'
]

#Meteorological features
meteo_features = ['temp_max', 'temp_min', 'temp_mean', 'rain', 'humidity', 'wind_dir', 'wind_speed', 'wind_gusts', 'rad', 'sun_hours', 'pressure']

#Defining new input Features

#Defining the different time windows (1/2 weeks, 1/3/6 months)
time_windows = {
    '1w': 7,
    '2w': 14,
    '1m': 30,
    '3m': 90,
    '6m': 180
}

SPLIT_YEAR = 2019
END_YEAR = 2020

# Replace all '--' occurrences with previous day values
data.replace('--', np.nan, inplace=True)
data.replace('', np.nan, inplace=True)
data.replace(' ', np.nan, inplace=True)
data.ffill(inplace=True)

# change all columns to float except datetime
for column in data.columns:
    if column not in ['datetime']:
        data[column] = data[column].astype('float32')

# Fill empty values with the previous day values

#Creating rolling mean and variance features for each feature in the given time windows
new_features = {}

for feature in taxa + meteo_features:
    for window_name, window_size in time_windows.items():
        #Rolling mean
        new_features[f'{feature}_rolling_mean_{window_name}'] = data[feature].rolling(window=window_size, min_periods=1).mean()
        #Rolling variance
        new_features[f'{feature}_rolling_var_{window_name}'] = data[feature].rolling(window=window_size, min_periods=1).var()

new_features_df = pd.DataFrame(new_features)

data = pd.concat([data, new_features_df], axis=1)

#Dropping possible existing rows with NaN values created by shifts and rolling sums
data.dropna(inplace=True)


#Random Forests training and tuning
#Training a RF for each pollen type
for taxon in taxa:
    #Defining the final feature set to use
    #Here, we still use year, month and day as feature -> TRY TO NOT INCLUDE THEM AND COMPARE THE RESULTS
    features = ['temp_max', 'temp_min', 'temp_mean', 'rain', 'humidity', 'wind_dir', 'wind_speed', 'wind_gusts', 'rad', 'sun_hours', 'pressure'] + [f'{taxon}_rolling_mean_{window_name}' for window_name in time_windows] + [f'{taxon}_rolling_var_{window_name}' for window_name in time_windows] + [f'{feature}_rolling_mean_{window_name}' for feature in meteo_features for window_name in time_windows] + [f'{feature}_rolling_var_{window_name}' for feature in meteo_features for window_name in time_windows]

    #Target feature -> rolling mean for the next time window (CHANGE VALUES BELOW TO CHANGE TIME WINDOW)
    tw_name = '1w' #PREDICTING ONE WEEK AHEAD
    tw_size = 7

    # center=False means that the center is the rightmost element of the window
    data[f'{taxon}_target_{tw_name}'] = (data[taxon].shift(-tw_size).rolling(window=tw_size, min_periods=1, center=False, closed='right').mean())

    #Ensuring time series consistency for the splits by filtering dates
    train_data = data[data['year'] <= SPLIT_YEAR] #Training on data up to 2015
    test_data = data[(data['year'] > SPLIT_YEAR) & (data['year'] <= END_YEAR)] #Testing on 2016-2020 data


    #Initializing TimeSeriesSplit, keeping consistent splits
    tscv = TimeSeriesSplit(n_splits=5)

    # cuML

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = train_data[features]
    y_train = train_data[f'{taxon}_target_{tw_name}']
    X_test = test_data[features]
    y_test = test_data[f'{taxon}_target_{tw_name}']

    # from sklearn.model_selection import train_test_split
    # X = data[features]
    # y = data[f'{taxon}_target_{tw_name}']
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = pd.DataFrame(X_train).astype('float32')
    X_test = pd.DataFrame(X_test).astype('float32')
    y_train = pd.DataFrame(y_train).astype('float32')

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print("Checking for NaN and Inf values in the training and testing data for taxon: ", taxon)
    # print(np.isnan(X_train).sum(), np.isinf(X_train).sum())
    # print(np.isnan(X_test).sum(), np.isinf(X_test).sum())
    #
    # print(np.isnan(y_train).sum(), np.isinf(y_train).sum())
    # print(np.isnan(y_test).sum(), np.isinf(y_test).sum())

    params = {
        'n_estimators': 800,
        'split_criterion': 'mse',
        'bootstrap': True,
        'verbose': 0,
        'output_type': 'input'
    }

    param_grid = {
        'max_depth': [5, 7, 10],
        'n_bins': [128, 256],
    }

    rf = cuml_rf.RandomForestRegressor(**params)
    rf_random = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_


    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Taxon: {taxon}")
    print(f"Tuned R² Score: {r2:.4f}") #Higher is better; measure of how well the model explains variance in the test data
    print(f"Tuned Mean Squared Error: {mse:.4f}\n") #Lower is better; Average squared difference between predicted and actual values

    #Showing each feature importance for future feature tuning/adjustments
    # feature_importances = best_rf.feature_importances_
    # sns.barplot(x=feature_importances, y=features)
    # plt.title(f"Feature Importance for {taxon}")
    # plt.savefig(f'./plots/{taxon}_{tw_name}_fi.png')

    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.plot(test_data['datetime'], y_test, color='green', label='Actual')
    ax.plot(test_data['datetime'], y_pred, color='red', alpha=0.5, label='Predicted')
    ax.grid()
    fig.legend()
    plt.savefig(f'./plots/{taxon}_{tw_name}_pred.png')
