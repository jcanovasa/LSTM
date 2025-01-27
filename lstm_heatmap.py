import os
import logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import keras_tuner as kt
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp'])
    data['timestamp'] = data['timestamp'].astype(int)
    data['timestamp'] = data['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['value']])
    return data, scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(50, 1)))  # Input shape for sequences of length 50
    
    # Add LSTM layers with tunable number of neurons and layers
    for i in range(hp.Int('num_layers', 1, 3)):  # 1 to 3 layers
        model.add(LSTM(
            units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32),  # Tunable neurons per layer
            return_sequences=(i < hp.Int('num_layers', 1, 3) - 1)  # Return sequences for all but the last layer
        ))
    
    # Add Dense output layer
    model.add(Dense(1))
    
    # Compile model with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])  # Tunable learning rate
        ),
        loss='mean_squared_error'
    )
    return model

# Train and evaluate the model using Keras Tuner
def train_with_keras_tuner(file_path, max_trials=10, executions_per_trial=1):
    data, scaled_data, scaler = load_and_preprocess_data(file_path)
    
    # Create sequences
    sequence_length = 50
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create a tuner instance
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,  # Number of hyperparameter combinations to try
        executions_per_trial=executions_per_trial,  # Average results over multiple runs
        directory='tuner_results',
        project_name='lstm_hyperparameter_tuning'
    )
    
    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,  # Number of epochs for each trial
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )
    
    # Get the best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best number of layers: {best_hps.get('num_layers')}")
    for i in range(best_hps.get('num_layers')):
        print(f"Best units for layer {i}: {best_hps.get('units_' + str(i))}")
    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    
    # Train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )
    
    # Evaluate the best model
    predictions = best_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Unscale predictions
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Unscale actual values
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_rescaled - predictions) ** 2))
    print(f"RMSE: {rmse}")
    
    # Save predictions and actual values to CSV
    guess_data = pd.DataFrame({'ds': data['timestamp'].iloc[-len(y_test):].values, 'yhat': predictions.flatten()})
    eval_data = pd.DataFrame({'ds': data['timestamp'].iloc[-len(y_test):].values, 'y': y_test_rescaled.flatten()})
    guess_data.to_csv('guess_data.csv', index=False)
    eval_data.to_csv('eval_data.csv', index=False)
    
    return history, rmse

# Example: Train and tune with Keras Tuner
if __name__ == "__main__":
    file_path = "data.csv"
    print("Starting hyperparameter tuning...")
    history, rmse = train_with_keras_tuner(file_path)
    print(f"Tuning complete. Best model RMSE: {rmse}")
