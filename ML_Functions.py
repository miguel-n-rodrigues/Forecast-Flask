import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import itertools
from math import sqrt
from sklearn.metrics import mean_squared_error
import os



###################################
#                                 #
#   Data Manipulation Functions   #
#                                 #
###################################

# Cross-validation: rolling window for test
def moving_window_test(data, training_period, testing_period):
    df_trn = data.iloc[:training_period + testing_period, ].copy()
    df_trn["key"] = "training"

    df_tst = df_trn.tail(testing_period).copy()
    df_tst["key"] = "test"

    df_trn = df_trn.iloc[:-testing_period]

    return df_trn, df_tst


# Cross-validation: rolling window for validation
def moving_window_val(training_period, testing_period, val_weeks, df_trn):
    df_trn_val = df_trn.iloc[:training_period - val_weeks * testing_period].copy()

    df_val = df_trn.iloc[
             training_period - val_weeks * testing_period:training_period - (val_weeks - 1) * testing_period, :].copy()
    df_val["key"] = "validation"

    return df_trn_val, df_val


# Create df with every combination
def expand_grid(dct):
    rows = itertools.product(*dct.values())

    return pd.DataFrame.from_records(rows, columns=dct.keys())


# Divide into train, val and test, normalize, create shifts
def data_processing(data_train, data_val_test, tsteps, know_on_day_vars=5):
    # Combine into 1
    df = pd.concat([data_train, data_val_test], axis=0)

    # Normalization

    # Features' name
    df_colnames = list(df.columns)
    sety = {"index", "key"}
    features = [o for o in df_colnames if o not in sety]

    # Number of features
    feature_number = len(features)

    # Min and Max values
    min_max_array = np.zeros((feature_number, 2))  # rows= features, colnames= c("min","max")
    min_max_array[:, 0] = data_train.drop(['index', 'key'],
                                          axis=1).min()  # drops the "index" and "key" columns, and computes the min values for the training period
    min_max_array[:, 1] = data_train.drop(['index', 'key'],
                                          axis=1).max()  # drops the "index" and "key" columns, and computes the max values for the training period

    # Normalize
    df_processed_tbl = df.copy()
    columns_to_convert = df_processed_tbl.columns[1:feature_number + 1]  # Select columns 1 through 10
    df_processed_tbl[columns_to_convert] = df_processed_tbl[columns_to_convert].astype("float64")

    for x in range(1, feature_number + 1):
        df_processed_tbl.iloc[:, x] = (df.iloc[:, x] - min_max_array[x - 1, 0]) / (
                min_max_array[x - 1, 1] - min_max_array[x - 1, 0])

    # Lags
    know_on_day_vars = know_on_day_vars  # counting from left to right including the depedent var
    lag_tbl = np.zeros((len(df_processed_tbl), (feature_number) * (tsteps + 1)))

    for f in range(1, feature_number + 1):  # Save the feature values
        lag_tbl[:, (tsteps + 1) * (f - 1)] = df_processed_tbl.iloc[:, f]

        for i in range(1, tsteps + 1):  # Create feature lags
            lag_tbl[:, (tsteps + 1) * (f - 1) + i] = df_processed_tbl.iloc[:, f].shift(i)

        # Delete first shift lines since they now have NAs
    lag_tbl = np.delete(lag_tbl, [range(0, tsteps)], axis=0)

    # Remove variables that you don't know on the day and the last shift if you do
    sequence_of_numbers = []

    # Do not know on the day
    numbers = range(tsteps + 1, (tsteps + 1) * know_on_day_vars, tsteps + 1)
    for number in numbers:
        sequence_of_numbers.append(number)

        # Do know on the day
    sequence_of_numbers_known = []
    numbers = range((tsteps + 1) * (know_on_day_vars), np.shape(lag_tbl)[1], tsteps + 1)
    for number in numbers:
        sequence_of_numbers_known.append(number)
    x = [i for i in range((tsteps + 1) * (know_on_day_vars), np.shape(lag_tbl)[1]) if
         i not in sequence_of_numbers_known]
    sequence_of_numbers = np.concatenate([sequence_of_numbers, x])

    # Remove them
    lag_tbl = np.delete(lag_tbl, sequence_of_numbers, axis=1)

    return lag_tbl, feature_number, min_max_array


# Transform train, val, and test into 2D array format
def array_transformation_2D(lag_tbl, testing_period):
    row_number = np.shape(lag_tbl)[0]
    train, val_test = lag_tbl[:-testing_period], lag_tbl[row_number - testing_period:row_number]

    # Train
    # x
    x_train = np.delete(train, 0, axis=1)
    # y
    y_train = train[:, 0]

    # Val/Test
    # x
    x_val_test = np.delete(val_test, 0, axis=1)
    # y
    y_val_test = val_test[:, 0]

    return x_train, y_train, x_val_test, y_val_test


def data_processing_3D_array(data_train, data_val_test, tsteps, testing_period, know_on_day_vars=5):
    # Combine into 1
    df = pd.concat([data_train, data_val_test], axis=0)

    # Normalization

    # Features' name
    df_colnames = list(df.columns)
    sety = {"index", "key"}
    features = [o for o in df_colnames if o not in sety]

    # Number of features
    feature_number = len(features)

    # Min and Max values
    min_max_array = np.zeros((feature_number, 2))
    min_max_array[:, 0] = data_train.drop(['index', 'key'],
                                          axis=1).min()
    min_max_array[:, 1] = data_train.drop(['index', 'key'],
                                          axis=1).max()

    # Normalize
    df_processed_tbl = df.copy()
    columns_to_convert = df_processed_tbl.columns[1:feature_number + 1]  # Select columns 1 through 10
    df_processed_tbl[columns_to_convert] = df_processed_tbl[columns_to_convert].astype("float64")

    for x in range(1, feature_number + 1):
        df_processed_tbl.iloc[:, x] = (df.iloc[:, x] - min_max_array[x - 1, 0]) / (
                min_max_array[x - 1, 1] - min_max_array[x - 1, 0])

    # Lags
    know_on_day_vars = know_on_day_vars
    lag_tbl = np.zeros((len(df_processed_tbl), (feature_number) * (tsteps + 1)))

    for f in range(1, feature_number + 1):
        lag_tbl[:, (tsteps + 1) * (f - 1)] = df_processed_tbl.iloc[:, f]

        for i in range(1, tsteps + 1):
            lag_tbl[:, (tsteps + 1) * (f - 1) + i] = df_processed_tbl.iloc[:, f].shift(i)

        # Delete first shift lines since they now have NAs
    lag_tbl = np.delete(lag_tbl, [range(0, tsteps)], axis=0)

    # Remove variables that you don't know on the day and the last shift if you do
    sequence_of_numbers = []

    # Do not know on the day
    numbers = range(tsteps + 1, (tsteps + 1) * know_on_day_vars, tsteps + 1)
    for number in numbers:
        sequence_of_numbers.append(number)

        # Do know on the day
    sequence_of_numbers_known = []
    numbers = range((tsteps + 1) * (know_on_day_vars), np.shape(lag_tbl)[1], tsteps + 1)
    for number in numbers:
        sequence_of_numbers_known.append(number)
    x = [i for i in range((tsteps + 1) * (know_on_day_vars), np.shape(lag_tbl)[1]) if
         i not in sequence_of_numbers_known]
    sequence_of_numbers = np.concatenate([sequence_of_numbers, x])

    # Remove them
    lag_tbl = np.delete(lag_tbl, sequence_of_numbers, axis=1)

    # Prepare for 3D array Transformation
    # Number of features
    feature_number = lag_tbl.shape[1]

    # Add lags to every var to feed to LSTM
    lag_tbl = pd.DataFrame(lag_tbl)

    lag_tbl1 = np.zeros((len(lag_tbl), (feature_number - 1) * tsteps + 1))
    lag_tbl1[:, 0] = lag_tbl.iloc[:, 0]

    for f in range(1, feature_number):  # Save the feature values
        lag_tbl1[:, (tsteps) * (f) - (tsteps-1)] = lag_tbl.iloc[:, f]

        for i in range(1, tsteps):  # Create feature lags
            lag_tbl1[:, (tsteps) * (f - 1) + i + 1] = lag_tbl.iloc[:, f].shift(i)

    lag_tbl = lag_tbl1.copy()
    del lag_tbl1

    # Delete first shift lines since they now have NAs
    lag_tbl = np.delete(lag_tbl, [range(0, tsteps)], axis=0)

    # 3D array Transformation
    row_number = np.shape(lag_tbl)[0]
    train, val_test = lag_tbl[:-tsteps], lag_tbl[row_number - tsteps:row_number]

    # Train
    # x
    x_train_vec = np.delete(train, 0, axis=1)

    # 2D array
    x_train = np.transpose(x_train_vec[0, :].reshape(-1, tsteps))  # first line
    for array_number in range(1, len(x_train_vec)):  # remaining lines
        x_train = np.concatenate((x_train,
                                  np.transpose(x_train_vec[(array_number), :].reshape(-1, tsteps))
                                  ), axis=1)
        # 3D array
    x_train = np.array_split(x_train, len(x_train_vec), axis=1)
    x_train = np.reshape(x_train, (len(x_train_vec), tsteps, int(x_train_vec.shape[1] / tsteps)))

    # y
    y_train = train[:, 0]
    y_train = np.reshape(y_train, (row_number - tsteps, 1))
    y_train = y_train.flatten()

    # Val/Test
    # x
    x_val_test_vec = np.delete(val_test, 0, axis=1)
    # 2D array
    x_val_test = np.transpose(x_val_test_vec[0, :].reshape(-1, tsteps))  # first line
    for array_number in range(1, testing_period):  # remaining lines
        x_val_test = np.concatenate((x_val_test,
                                     np.transpose(x_val_test_vec[(array_number), :].reshape(-1, testing_period))
                                     ), axis=1)
        # 3D array
    x_val_test = np.array_split(x_val_test, testing_period, axis=1)
    x_val_test = np.reshape(x_val_test, (testing_period, tsteps, int(x_train_vec.shape[1] / tsteps)))

    # y
    y_val_test = val_test[:, 0]
    y_val_test = np.reshape(y_val_test, (tsteps, 1))
    y_val_test = y_val_test.flatten()

    return lag_tbl, feature_number, min_max_array, x_train, y_train, x_val_test, y_val_test


###################################
#                                 #
#   Random Forest Training Model  #
#                                 #
###################################
def RF_prediction_val(data_x, data_y, model, min_max_array):
    yhat_val = model.predict(data_x)

    # Invert scaling for forecast
    yhat_val = yhat_val.flatten()
    yhat_val = yhat_val * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    yhat_val = np.around(yhat_val, 0)

    # Invert scaling for actual
    y_val = data_y * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    y_val = np.around(y_val, 0)

    return yhat_val, y_val


def RF_prediction_test(data_x, data_y, model, min_max_array):
    yhat = model.predict(data_x)

    # Invert scaling for forecast
    yhat = yhat.flatten()
    yhat = yhat * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    yhat = np.around(yhat, 0)

    # Invert scaling for actual
    y = data_y * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    y = np.around(y, 0)

    rmse = sqrt(mean_squared_error(y, yhat))
    print('Test RMSE: %.3f' % rmse)

    return yhat, y


def RF_Forecast(data, training_period, testing_period, number_val_months, save_folder="NA"):
    print("Starting Random Forest Prediction")

    # Cross-validation
    training_period = training_period  # train until end of 2019
    testing_period = testing_period  # number of instances ahead

    # Divide into train and test, standardize, create shifts
    shifts = testing_period
    number_val_months = number_val_months

    # RandomForest training model
    Combination_grid = expand_grid({'max_features': np.round(np.linspace(0.1, 1, 10), 2),
                                    'n_estimators': range(100, 1000 + 1, 100)})

    loop_number = 1

    # Outer loop
    while training_period < (len(data)):

        # Cross-validation
        df_trn, df_tst = moving_window_test(data, training_period, testing_period)

        # RandomForest training model

        # Parameters and RMSE matrix
        Inner_Loop_Metrics = np.zeros((len(Combination_grid), len(Combination_grid.columns) + 1))
        Inner_Loop_Metrics = pd.DataFrame(Inner_Loop_Metrics, columns=['max_features', 'n_estimators', 'RMSE'])
        Inner_Loop_Metrics_row = 0

        # Inner loop
        for combinations in range(0, len(Inner_Loop_Metrics), 1):
            if combinations % 10 == 0:
                print("testing combination no.", combinations + 1, "/", len(Inner_Loop_Metrics))

            for val_weeks in range(1, number_val_months + 1):

                df_trn_val, df_val = moving_window_val(training_period, testing_period, val_weeks, df_trn)

                lag_tbl, feature_number, min_max_array = data_processing(data_train=df_trn_val, data_val_test=df_val, tsteps=shifts)

                x_train, y_train, x_val_test, y_val_test = array_transformation_2D(lag_tbl, shifts)

                # parameters
                rf = RandomForestRegressor(max_features=Combination_grid.iloc[combinations, 0],
                                           n_estimators=int(Combination_grid.iloc[combinations, 1])
                                           )

                # train
                rf.fit(x_train, y_train)

                # validate
                yhat_val, y_val = RF_prediction_val(data_x=x_val_test, data_y=y_val_test, model=rf, min_max_array=min_max_array)

                # save forecast value
                if 'fc_values' in globals():
                    fc_values = np.append(fc_values, yhat_val)
                else:
                    fc_values = yhat_val.copy()
                # save real values
                if 'real_values' in globals():
                    real_values = np.append(real_values, y_val)
                else:
                    real_values = y_val.copy()

            rmse = sqrt(mean_squared_error(real_values, fc_values))

            # record
            Inner_Loop_Metrics.iloc[Inner_Loop_Metrics_row] = Combination_grid.iloc[combinations, 0], Combination_grid.iloc[
                combinations, 1], rmse

            Inner_Loop_Metrics_row += 1

            del fc_values, real_values

        # RandomForest forecast

        # select best pair of parameters
        max_features = Inner_Loop_Metrics.iloc[Inner_Loop_Metrics['RMSE'].idxmin(), 0]
        n_estimators = int(Inner_Loop_Metrics.iloc[Inner_Loop_Metrics['RMSE'].idxmin(), 1])

        rf = RandomForestRegressor(max_features=max_features,
                                   n_estimators=n_estimators
                                   )

        lag_tbl, feature_number, min_max_array = data_processing(data_train=df_trn, data_val_test=df_tst, tsteps=shifts)

        x_train, y_train, x_val_test, y_val_test = array_transformation_2D(lag_tbl, shifts)
    
        # train
        rf.fit(x_train, y_train)
    
        # predict
        yhat, y = RF_prediction_test(data_x=x_val_test, data_y=y_val_test, model=rf, min_max_array=min_max_array)
    
        if 'all_yhat' in globals():
            all_yhat1 = pd.concat([df_tst["index"].reset_index(drop=True), pd.DataFrame(yhat.copy()).reset_index(drop=True)], axis=1)
            all_yhat = pd.concat([all_yhat, all_yhat1], axis=0)
        else:
            all_yhat = pd.concat([df_tst["index"].reset_index(drop=True), pd.DataFrame(yhat.copy()).reset_index(drop=True)], axis=1)
    
        if 'optimal_parameters' in globals():
            optimal_parameters1 = np.zeros((1, len(Combination_grid.columns)))
            optimal_parameters1 = pd.DataFrame(optimal_parameters1, columns=['max_features', 'n_estimators'])
            optimal_parameters1.iloc[0, :] = max_features, n_estimators
            optimal_parameters = optimal_parameters.append(optimal_parameters1)
        else:
            optimal_parameters = np.zeros((1, len(Combination_grid.columns)))
            optimal_parameters = pd.DataFrame(optimal_parameters, columns=['max_features', 'n_estimators'])
            optimal_parameters.iloc[0, :] = max_features, n_estimators
    
        training_period += testing_period
    
        print("Loop no.", int(loop_number))
    
        loop_number += 1

    df_all_yhat = pd.DataFrame(all_yhat)

    df_optimal_parameters = pd.DataFrame(optimal_parameters)

    if save_folder != "NA":
        filepath = (save_folder + "/RF_all_yhat.csv").replace("\\", "/")
        df_all_yhat.to_csv(os.path.normpath(filepath))

        filepath = (save_folder + "/RF_optimal_parameters.csv").replace("\\", "/")
        df_optimal_parameters.to_csv(os.path.normpath(filepath))

    print("finish")

    return df_all_yhat, df_optimal_parameters


##########################
#                        #
#   LSTM Training Model  #
#                        #
##########################


def LSTM_model_stateless_train_val(x_data, y_data, batch_size, epochs, units1, units2):
    # design network
    model = Sequential()
    model.add(
        LSTM(units1,
             batch_input_shape=(batch_size, x_data.shape[1], x_data.shape[2]),
             return_sequences=True)
    )
    model.add(
        LSTM(units2,
             return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    # fit network
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=[Callback1()],
                        shuffle=False)

    return model, history


def LSTM_model_stateless_train_test(x_data, y_data, batch_size, epochs, units1, units2):
    # design network
    model = Sequential()
    model.add(
        LSTM(units1,
             batch_input_shape=(batch_size, x_data.shape[1], x_data.shape[2]),
             return_sequences=True))
    model.add(
        LSTM(units2,
             return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    # fit network
    history = model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=[Callback()],
                        shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.legend()
    pyplot.show()

    return model, history


# Callbacks
class Callback1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 300 == 0:
            print("The average loss on epoch {} is {:7.2f} and mean squared error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_squared_error"]
            )
            )


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print("The average loss on epoch {} is {:7.2f} and mean squared error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_squared_error"]
            )
            )


# LSTM prediction
def LSTM_prediction_val(data_x, data_y, model, min_max_array):
    yhat_val = model.predict(data_x)

    # invert scaling for forecast
    yhat_val = yhat_val.flatten()
    yhat_val = yhat_val * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    yhat_val = np.around(yhat_val, 0)

    # invert scaling for actual
    y_val = data_y * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    y_val = np.around(y_val, 0)

    return yhat_val, y_val


def LSTM_prediction_test(data_x, data_y, model, min_max_array):
    yhat = model.predict(data_x)

    # invert scaling for forecast
    yhat = yhat.flatten()
    yhat = yhat * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    yhat = np.around(yhat, 0)

    # invert scaling for actual
    y = data_y * (min_max_array[0, 1] - min_max_array[0, 0]) + min_max_array[0, 0]
    y = np.around(y, 0)

    rmse = sqrt(mean_squared_error(y, yhat))
    print('Test RMSE: %.3f' % rmse)

    return yhat, y


def LSTM_Forecast(data, training_period, testing_period, number_val_months, batch_size, save_folder="NA"):
    print("Starting LSTM Prediction")

    # Cross-validation
    training_period = training_period  # train until end of 2019
    testing_period = testing_period  # number of instances ahead

    # Divide into train and test, standardize, create shifts
    shifts = testing_period
    number_val_months = number_val_months
    batch_size = batch_size

    # LSTM training model
    Combination_grid = expand_grid({'neurons_first_layer': np.array([10, 50, 100]),
                                    'neurons_second_layer': np.array([10, 50, 100])
                                    })

    loop_number = 1

    # Outer loop
    while training_period < (len(data)):

        # cross-validation
        df_trn, df_tst = moving_window_test(data, training_period, testing_period)

        # LSTM training model

        # Parameters and RMSE matrix
        Inner_Loop_Metrics = np.zeros((len(Combination_grid), len(Combination_grid.columns) + 1))
        Inner_Loop_Metrics = pd.DataFrame(Inner_Loop_Metrics, columns=['neurons_first_layer', 'neurons_second_layer', 'RMSE'])
        Inner_Loop_Metrics_row = 0

        # Inner loop
        for combinations in range(0, len(Inner_Loop_Metrics), 1):
            print("testing combination no.", combinations + 1, "/", len(Inner_Loop_Metrics))

            for val_weeks in range(1, number_val_months + 1):

                df_trn_val, df_val = moving_window_val(training_period, testing_period, val_weeks, df_trn)

                lag_tbl, FeatureNumber, min_max_array, x_train, y_train, x_val_test, y_val_test = data_processing_3D_array(data_train=df_trn_val, data_val_test=df_val, tsteps=shifts, testing_period=testing_period)


                # train
                model, history = LSTM_model_stateless_train_val(x_data=x_train, y_data=y_train, batch_size=batch_size, epochs=100,
                                                                units1=int(Combination_grid.iloc[combinations, 0]),
                                                                units2=int(Combination_grid.iloc[combinations, 1])
                                                                )

                # validate
                yhat_val, y_val = LSTM_prediction_val(data_x=x_val_test, data_y=y_val_test, model=model, min_max_array=min_max_array)

                # save forecast value
                if 'fc_values' in globals():
                    fc_values = np.append(fc_values, yhat_val)
                else:
                    fc_values = yhat_val.copy()
                # save real values
                if 'real_values' in globals():
                    real_values = np.append(real_values, y_val)
                else:
                    real_values = y_val.copy()

            rmse = sqrt(mean_squared_error(real_values, fc_values))

            # record
            Inner_Loop_Metrics.iloc[Inner_Loop_Metrics_row] = Combination_grid.iloc[combinations, 0], Combination_grid.iloc[
                combinations, 1], rmse

            Inner_Loop_Metrics_row += 1

            del fc_values, real_values

        # LSTM forecast

        # select best pair of neurons
        neurons_first_layer = int(Inner_Loop_Metrics.iloc[Inner_Loop_Metrics['RMSE'].idxmin(), 0])
        neurons_second_layer = int(Inner_Loop_Metrics.iloc[Inner_Loop_Metrics['RMSE'].idxmin(), 1])

        lag_tbl, FeatureNumber, min_max_array, x_train, y_train, x_val_test, y_val_test = data_processing_3D_array(data_train=df_trn, data_val_test=df_tst, tsteps=shifts, testing_period=testing_period)

        # train
        model, history = LSTM_model_stateless_train_test(x_data=x_train, y_data=y_train, batch_size=batch_size, epochs=100,
                                                         units1=neurons_first_layer,
                                                         units2=neurons_second_layer)

        # predict
        yhat, y = LSTM_prediction_test(data_x=x_val_test, data_y=y_val_test, model=model, min_max_array=min_max_array)

        if 'all_yhat' in globals():
            all_yhat1 = pd.concat([df_tst["index"].reset_index(drop=True), pd.DataFrame(yhat.copy()).reset_index(drop=True)], axis=1)
            all_yhat = pd.concat([all_yhat, all_yhat1], axis=0, ignore_index=True)
        else:
            all_yhat = pd.concat([df_tst["index"].reset_index(drop=True), pd.DataFrame(yhat.copy()).reset_index(drop=True)], axis=1)

        if 'optimal_parameters' in globals():
            optimal_parameters1 = np.zeros((1, len(Combination_grid.columns)))
            optimal_parameters1 = pd.DataFrame(optimal_parameters1, columns=['neurons_first_layer', 'neurons_second_layer'])
            optimal_parameters1.iloc[0, :] = neurons_first_layer, neurons_second_layer
            optimal_parameters = pd.concat([optimal_parameters, optimal_parameters1], axis=0, ignore_index=True)
        else:
            optimal_parameters = np.zeros((1, len(Combination_grid.columns)))
            optimal_parameters = pd.DataFrame(optimal_parameters, columns=['neurons_first_layer', 'neurons_second_layer'])
            optimal_parameters.iloc[0, :] = neurons_first_layer, neurons_second_layer

        training_period += testing_period

        print("Loop no.", int(loop_number))

        loop_number += 1

    df_all_yhat = pd.DataFrame(all_yhat)

    df_optimal_parameters = pd.DataFrame(optimal_parameters)

    if save_folder != "NA":
        filepath = (save_folder + "/LSTM_all_yhat.csv").replace("\\", "/")
        df_all_yhat.to_csv(os.path.normpath(filepath))

        filepath = (save_folder + "/LSTM_optimal_parameters.csv").replace("\\", "/")
        df_optimal_parameters.to_csv(os.path.normpath(filepath))

    print("finish")

    return df_all_yhat, df_optimal_parameters
