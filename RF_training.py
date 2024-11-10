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
meteo_features = ['temp_max', 'temp_min', 'temp_mean', 'precipitation']

#Defining new input Features
#Creating lagged and trend features for both meteo and concentration features
for feature in taxa + meteo_features:
    data[f'{feature}_lag7'] = data[feature].shift(7)
    data[f'{feature}_trend7'] = data[feature] - data[f'{feature}_lag7']

#Creating 7-day sum feature for precipitation
data['precipitation_sum7'] = data['precipitation'].rolling(window=7).sum()

#Dropping possible existing rows with NaN values created by shifts and rolling sums
data.dropna(inplace=True)

#Defining the final feature set to use
features = ['year', 'month', 'day', 'temp_max', 'temp_min', 'temp_mean', 'precipitation', 'precipitation_sum7'] + [f'{feature}_lag7' for feature in taxa + meteo_features] + [f'{feature}_trend7' for feature in taxa + meteo_features]

#Random Forests training and tuning
#Training a RF for each pollen type
for taxon in taxa:
    #Target feature -> pollen concentration of the following week (->  -7-days lagged taxa concentration)
    data[f'{taxon}_target'] = data[taxon].shift(-7)
    data.dropna(subset=[f'{taxon}_target'], inplace=True)
    
    #Ensuring time series consistency for the splits by filtering dates
    train_data = data[data['year'] <= 2021]  #Training on data up to 2021
    test_data = data[data['year'] == 2022]   #Testing on 2022 data
    
    X_train = train_data[features]
    y_train = train_data[f'{taxon}_target']
    X_test = test_data[features]
    y_test = test_data[f'{taxon}_target']

    # Initialize TimeSeriesSplit, keeping consistent splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    #Parameters grid used to look for the most fitting max_depth parameter
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
    }

    #Random Forest here created with no hyperparameters (such as max_depth) because of the following tuning process
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=tscv, verbose=2, n_jobs=-1)
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
    plt.show()

    #Results visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title(f"Predicted vs Actual for {taxon}")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.plot(test_data['datetime'], y_test, color='green', label='real')
    ax.plot(test_data['datetime'], y_pred, color='red', label='predicted') 
    ax.grid()  
    fig.legend() 
    plt.show() 
    plt.close(fig) 
