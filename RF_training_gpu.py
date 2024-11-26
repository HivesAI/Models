# Libraries for data processing
import numpy as np
import pandas as pd

# Library for plotting
import matplotlib.pyplot as plt

# Libraries for model training and validation
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import cuml.ensemble.randomforestregressor as cuml_rf

# Data path
data_path = './data/Full_data_Trentino.csv'

# Loading data
data = pd.read_csv(data_path)
data['datetime'] = pd.to_datetime(data['datetime'])

data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.dayofyear

# Taxa concentrations
taxa = [
    'Ambrosia', 'Artemisia', 'Betula', 'Corylus', 'Cupressaceae, Taxaceae',
    'Fraxinus', 'Olea europaea', 'Ostrya carpinifolia', 'Poaceae', 'Urticaceae'
]

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

# Fill empty values with the previous day values

# Creating rolling mean and variance features for each feature in the given time windows
new_features = {}

for feature in taxa + meteo_features:
    for window_name, window_size in time_windows.items():
        # Rolling mean
        new_features[f'{feature}_rolling_mean_{window_name}'] = data[feature].rolling(window=window_size, min_periods=1).mean()
        # Rolling variance
        new_features[f'{feature}_rolling_var_{window_name}'] = data[feature].rolling(window=window_size, min_periods=1).var()


for feature in taxa + meteo_features:
    for window_name, window_size in time_windows.items():
        new_features[f'{feature}_rolling_mean_{window_name}_delta'] = new_features[f'{feature}_rolling_mean_{window_name}'] - new_features[f'{feature}_rolling_mean_{window_name}'].shift(window_size)
        new_features[f'{feature}_rolling_var_{window_name}_delta'] = new_features[f'{feature}_rolling_var_{window_name}'] - new_features[f'{feature}_rolling_var_{window_name}'].shift(window_size)

        for i in range(1, 6):
            new_features[f'{feature}_rolling_mean_{window_name}_delta_{i}w'] = new_features[f'{feature}_rolling_mean_{window_name}'] - new_features[f'{feature}_rolling_mean_{window_name}'].shift(i)
            new_features[f'{feature}_rolling_var_{window_name}_delta_{i}w'] = new_features[f'{feature}_rolling_var_{window_name}'] - new_features[f'{feature}_rolling_var_{window_name}'].shift(i)


all_features = new_features.copy()
new_features_df = pd.DataFrame(new_features)

data = pd.concat([data, new_features_df], axis=1)

# Dropping possible existing rows with NaN values created by shifts and rolling sums
data.dropna(inplace=True)


SPLIT_YEAR = 2015
END_YEAR = 2020

res_file = open('results-gpu.txt', 'w')
res_file.write(f'Training years <= {SPLIT_YEAR} < Validation years <= {END_YEAR}\n')

# ALL_TAXA = True
# if ALL_TAXA:
#     print("All Taxa features")
# else:
#     print("Single taxon features")

for taxon in taxa:
    print(f"Training for {taxon}")

    features = ['temp_max', 'temp_min', 'temp_mean', 'rain', 'humidity', 'wind_dir', 'wind_speed', 'wind_gusts', 'rad', 'sun_hours', 'pressure'] + [f'{taxon}_rolling_mean_{window_name}' for window_name in time_windows] + [f'{taxon}_rolling_var_{window_name}' for window_name in time_windows] + [f'{feature}_rolling_mean_{window_name}' for feature in meteo_features for window_name in time_windows] + [f'{feature}_rolling_var_{window_name}' for feature in meteo_features for window_name in time_windows]

    features += [f'{taxon}_rolling_mean_{window_name}_delta' for window_name in time_windows] + [f'{taxon}_rolling_var_{window_name}_delta' for window_name in time_windows]

    for i in range(1, 6):
        features += [f'{taxon}_rolling_mean_{window_name}_delta_{i}w' for window_name in time_windows] + [f'{taxon}_rolling_var_{window_name}_delta_{i}w' for window_name in time_windows]


    # TODO: understand if this is actually correct
    # Target feature -> rolling mean for the next time window (CHANGE VALUES BELOW TO CHANGE TIME WINDOW)
    tw_name = '1w' # PREDICTING ONE WEEK AHEAD
    tw_size = 7

    data[f'{taxon}_target_{tw_name}'] = (data[taxon].shift(-tw_size).rolling(window=tw_size, min_periods=1, center=False, closed='right').mean())

    # Ensuring time series consistency for the splits by filtering dates
    train_data = data[data['year'] <= SPLIT_YEAR] # Training on data up to 2015
    test_data = data[(data['year'] > SPLIT_YEAR) & (data['year'] <= END_YEAR)] # Testing on 2016-2020 data


    # cuML

    tscv = TimeSeriesSplit(n_splits=5)
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

    params = {
        'n_estimators': 800,
        'split_criterion': 'mse',
        'bootstrap': True,
        'verbose': 0,
        'output_type': 'input'
    }

    param_grid = {
        'max_depth': [12],
        'n_bins': [512],
    }

    rf = cuml_rf.RandomForestRegressor(**params)
    rf_random = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_
    # for tree in rf.estimators_:
    #     tmp_pred = tree.predict(X_test)


    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Taxon: {taxon}")
    print(f"Tuned R² Score: {r2:.4f}") # Higher is better; measure of how well the model explains variance in the test data
    print(f"Tuned Mean Squared Error: {mse:.4f}\n") # Lower is better; Average squared difference between predicted and actual values

    res_file.write(f"Taxon: {taxon}\n")
    res_file.write(f"R2: {r2:.4f}\n")
    res_file.write(f"MSE: {mse:.4f}\n\n")


    # Showing each feature importance for future feature tuning/adjustments
    # feature_importances = best_rf.feature_importances_
    # sns.barplot(x=feature_importances, y=features)
    # plt.title(f"Feature Importance for {taxon}")
    # plt.savefig(f'./plots/{taxon}_{tw_name}_fi.png')


    # Instead of plotting the values for each day, plot the values for each week, where the value is the mean of the week
    # This is done to make the plot more readable
    weekly_y_test = []
    weekly_y_pred = []

    for i in range(0, len(y_test), 7):
        weekly_y_test.append(y_test[i:i+7].mean())
        weekly_y_pred.append(y_pred[i:i+7].mean())


    fig, ax = plt.subplots(1, figsize=(10,6))

    fig.suptitle(f'{taxon} - Predicted vs Actual for {tw_name}', fontsize=16)

    ax.plot(test_data['datetime'].iloc[::7], weekly_y_test, color='green', label='Actual')
    ax.fill_between(test_data['datetime'].iloc[::7], weekly_y_test, color='green', alpha=0.3)
    ax.plot(test_data['datetime'].iloc[::7], weekly_y_pred, color='red', label='Predicted')
    ax.fill_between(test_data['datetime'].iloc[::7], weekly_y_pred, color='red', alpha=0.3)

    ax.grid()
    fig.legend()
    plt.savefig(f'./plots/{taxon}_{tw_name}_pred.png')
    plt.close(fig)




res_file.close()
