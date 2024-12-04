# Libraries for data processing
import numpy as np
import pandas as pd

# Library for plotting
import matplotlib.pyplot as plt

# Libraries for model training and validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

import cuml.ensemble.randomforestregressor as cuml_rf
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from features_gen import get_features


SPLIT_YEAR = 2015
END_YEAR = 2020

xgb_res_file = open('results-xgb.txt', 'w')
xgb_res_file.write(f'Training years <= {SPLIT_YEAR} < Validation years <= {END_YEAR}\n')

cuml_res_file = open('results-cuml.txt', 'w')
cuml_res_file.write(f'Training years <= {SPLIT_YEAR} < Validation years <= {END_YEAR}\n')

best_res_file = open('best_results.txt', 'w')
best_res_file.write(f'Training years <= {SPLIT_YEAR} < Validation years <= {END_YEAR}\n')

taxa = [
    'Ambrosia', 'Artemisia', 'Betula', 'Corylus', 'Cupressaceae, Taxaceae',
    'Fraxinus', 'Olea europaea', 'Ostrya carpinifolia', 'Poaceae', 'Urticaceae'
]

# Taxa features to use for training for each species
species = {
    'Ambrosia': taxa,
    'Artemisia': ['Artemisia'],
    'Betula': ['Betula'],
    'Corylus': taxa,
    'Cupressaceae, Taxaceae': taxa,
    'Fraxinus': taxa,
    'Olea europaea': taxa,
    'Ostrya carpinifolia': ['Ostrya carpinifolia'],
    'Poaceae': ['Poaceae'],
    'Urticaceae': ['Urticaceae']
}


def save_results(taxon, model, file, r2, mse, y_test, y_pred, boosted=False):
    print(f"Taxon: {taxon}")
    # Higher is better; measure of how well the model explains variance in the test data
    print(f"R2: {r2:.4f}")
    # Lower is better; Average squared difference between predicted and actual values
    print(f"mse: {mse:.4f}\n")

    file.write(f"Taxon: {taxon}\n")
    file.write(f"R2: {r2:.4f}\n")
    file.write(f"mse: {mse:.4f}\n")

    if boosted:
        xgb_res_file.write(f"Boosting rounds: {model.num_boosted_rounds()}\n\n")

    # Instead of plotting the values for each day, plot the values for each week, where the value is the mean of the week
    weekly_y_test = []
    weekly_y_pred = []

    for i in range(0, len(y_test), 7):
        weekly_y_test.append(y_test[i:i+7].mean())
        weekly_y_pred.append(y_pred[i:i+7].mean())

    fig, ax = plt.subplots(1, figsize=(10, 6))

    fig.suptitle(f'{taxon} - Predicted vs Actual for {tw_name}', fontsize=16)

    ax.plot(test_data['datetime'].iloc[::7],
            weekly_y_test, color='green', label='Actual')
    ax.fill_between(test_data['datetime'].iloc[::7],
                    weekly_y_test, color='green', alpha=0.3)
    ax.plot(test_data['datetime'].iloc[::7],
            weekly_y_pred, color='red', label='Predicted')
    ax.fill_between(test_data['datetime'].iloc[::7],
                    weekly_y_pred, color='red', alpha=0.3)

    ax.grid()
    fig.legend()

    if boosted:
        plt.savefig(f'./plots-xgb/{taxon}_{tw_name}_pred.png')
    else:
        plt.savefig(f'./plots-cuml/{taxon}_{tw_name}_pred.png')



def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost")

    from sklearn.model_selection import train_test_split
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create DMatrix for validation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
      'objective': 'reg:squarederror',  # For regression
      'booster': 'gbtree',              # Tree booster
      'subsample': 1,                   # Use the entire dataset
      'colsample_bynode': 0.7,          # Subsampling features by node
      'max_depth': 5,                  # Maximum depth of each tree
      'learning_rate': 0.08,             # Contribution of each tree to the boosting step
      'device': 'cuda'
    }

    # Train the model
    num_round = 1500  # Number of boosting rounds
    # rf_model = xgb.train(params, dtrain, num_round)
    rf_model = xgb.train(params, dtrain, num_boost_round=num_round, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=20, verbose_eval=False)
    # Make predictions
    y_pred = rf_model.predict(dtest)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    save_results(taxon, rf_model, xgb_res_file, r2, mse, y_test, y_pred, boosted=True)
    return r2, mse



def train_cuml(X_train, y_train, X_test, y_test):
    print("Training cuML")

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

    y_pred = best_rf.predict(X_test)
    # Compute R2 score on non-zero values
    non_zero = (y_test > 0)
    r2 = r2_score(y_test[non_zero], y_pred[non_zero])
    mse = mean_squared_error(y_test, y_pred)

    save_results(taxon, best_rf, cuml_res_file, r2, mse, y_test, y_pred, boosted=False)

    return r2, mse



for taxon in taxa:
    print(f"\nTraining for {taxon}")

    data, features = get_features(species[taxon])
    print(f'Features={len(features)}\n')

    # Target feature -> rolling mean for the next time window (CHANGE VALUES BELOW TO CHANGE TIME WINDOW)
    tw_name = '1w'  # PREDICTING ONE WEEK AHEAD
    tw_size = 7

    data[f'{taxon}_target_{tw_name}'] = (data[taxon].shift(-tw_size).rolling(
        window=tw_size, min_periods=1, center=False, closed='right').mean())

    # Ensuring time series consistency for the splits by filtering dates
    # Training on data up to 2015
    train_data = data[data['year'] <= SPLIT_YEAR]
    test_data = data[(data['year'] > SPLIT_YEAR) & (data['year'] <= END_YEAR)]  # Testing on 2016-2020 data

    tscv = TimeSeriesSplit(n_splits=5)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = train_data[features]
    y_train = train_data[f'{taxon}_target_{tw_name}']
    X_test = test_data[features]
    y_test = test_data[f'{taxon}_target_{tw_name}']

    X_train = pd.DataFrame(X_train).astype('float32')
    X_test = pd.DataFrame(X_test).astype('float32')
    y_train = pd.DataFrame(y_train).astype('float32')

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select best results from either cumml or xgboost
    xgb_r2, xgb_mse = train_xgboost(X_train, y_train, X_test, y_test)
    cuml_r2, cuml_mse = train_cuml(X_train, y_train, X_test, y_test)

    if xgb_r2 > cuml_r2:
        best_res_file.write(f"Taxon: {taxon}\n")
        best_res_file.write(f"Method: XGBoost\n")
        best_res_file.write(f"R2: {xgb_r2:.4f}\n")
        best_res_file.write(f"MSE: {xgb_mse:.4f}\n\n")
    else:
        best_res_file.write(f"Taxon: {taxon}\n")
        best_res_file.write(f"Method: cuML\n")
        best_res_file.write(f"R2: {cuml_r2:.4f}\n")
        best_res_file.write(f"MSE: {cuml_mse:.4f}\n\n")



xgb_res_file.close()
cuml_res_file.close()
