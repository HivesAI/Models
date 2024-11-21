#THIS CODE AIMS TO TRAIN A RANDOM FOREST ON EACH TAXON WITHOUT HAVING OTHER TAXON CONCENTRATIONS AS INPUT FEATURES
#LOOK INTO "RF_training_ALL.py" FOR RF TRAINING ON EACH TAXON WHILE HAVING OTHER TAXON CONCENTRATIONS AS INPUT FEATURES

import os
#Libraries for data processing
import numpy as np
import pandas as pd

#Library for plotting
import matplotlib.pyplot as plt

#Libraries for model training and validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

#Directory to store results 
plots_dir = 'Plots_single_species_training'
os.makedirs(plots_dir,exist_ok=True)

#Data path
data_path = './data/Full_data_Trentino.csv'

#Loading data
data = pd.read_csv(data_path)
data['datetime'] = pd.to_datetime(data['datetime'])

data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.dayofyear

data.drop('datetime', axis=1, inplace=True)

data.replace('--', np.nan, inplace=True)
data.ffill(inplace=True)

# Defining features and taxa
time_features = ['year', 'month', 'day']
taxa = [col for col in (data.columns) if col[0].isupper()]  # Features starting with an uppercase letter
meteo_features = [col for col in data.columns if col not in time_features and col[0].islower()]  # Features starting with a lowercase letter
print("Taxa Features:", taxa)
print("Meteorological Features:", meteo_features)
#Defining new input Features

#Defining the different time windows (1/2 weeks, 1/3/6 months)
time_windows = {
    '1w': 7,
    '2w': 14,
    '1m': 30,
    '3m': 90,
    '6m': 180
}



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
    features = time_features + meteo_features + [f'{taxon}_rolling_mean_{window_name}' for window_name in time_windows] + [f'{taxon}_rolling_var_{window_name}' for window_name in time_windows] + [f'{feature}_rolling_mean_{window_name}' for feature in meteo_features for window_name in time_windows] + [f'{feature}_rolling_var_{window_name}' for feature in meteo_features for window_name in time_windows]
   
    #Target feature -> rolling mean for the next time window (CHANGE VALUES BELOW TO CHANGE TIME WINDOW)
    tw_name = '1w' #PREDICTING ONE WEEK AHEAD
    tw_size = 7

    data[f'{taxon}_target_{tw_name}'] = (data[taxon].shift(-7).rolling(window=tw_size, min_periods=1).mean())

    #Ensuring time series consistency for the splits by filtering dates
    train_data = data[data['year'] <= 2015] #Training on data up to 2015
    test_data = data[(data['year'] >= 2016) & (data['year'] <= 2020)] #Testing on 2016-2020 data

    X_train = train_data[features]
    y_train = train_data[f'{taxon}_target_{tw_name}']
    X_test = test_data[features]
    y_test = test_data[f'{taxon}_target_{tw_name}']

    #Initializing TimeSeriesSplit, keeping consistent splits
    tscv = TimeSeriesSplit(n_splits=5)

    #Parameters grid used to look for the most fitting max_depth parameter
    param_grid = {
        'max_depth': [2, 3, 5, 10, 12, None],
    }

    #Number of trees set at 500
    rf = RandomForestRegressor(n_estimators=500)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=6, cv=tscv, verbose=2, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_

    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Taxon: {taxon}")
    print(f"Tuned R² Score: {r2:.4f}") #Higher is better; measure of how well the model explains variance in the test data
    print(f"Tuned Mean Squared Error: {mse:.4f}\n") #Lower is better; Average squared difference between predicted and actual values

    #Showing each feature importance for future feature tuning/adjustments
    feature_importances = best_rf.feature_importances_
    sns.barplot(x=feature_importances, y=features)
    plt.title(f"Feature Importance for {taxon}")
    plt.savefig(f'{plots_dir}/{taxon}_feature_importance')
    #Results visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Rolling Mean of Concentration")
    plt.ylabel("Predicted Rolling Mean of Concentration")
    plt.title(f"Predicted vs Actual for {taxon}")
    plt.savefig(f'{plots_dir}/{taxon}_predicted_vs_actual')

    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.plot(test_data['datetime'], y_test, color='green', label='Actual')
    ax.plot(test_data['datetime'], y_pred, color='red', label='Predicted')
    ax.grid()
    fig.legend()
    plt.savefig(f'{plots_dir}/{taxon}_results')
    plt.close(fig)
