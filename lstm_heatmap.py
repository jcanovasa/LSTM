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

# Build the LSTM model dynamically based on hyperparameters
def build_lstm_model(sequence_length, neurons, layers):
    model = Sequential()
    # Add an explicit input layer
    model.add(Input(shape=(sequence_length, 1)))
    
    # Add LSTM layers
    for i in range(layers):
        if i == layers - 1:
            # Last LSTM layer should not return sequences
            model.add(LSTM(neurons, return_sequences=False))
        else:
            # Intermediate LSTM layers return sequences
            model.add(LSTM(neurons, return_sequences=True))
    
    # Add a Dense layer for output
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate the model
def train_and_evaluate_lstm_model(file_path, epochs, neurons, layers):
    data, scaled_data, scaler = load_and_preprocess_data(file_path)
    
    # Create sequences
    sequence_length = 50
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train the model
    model = build_lstm_model(sequence_length, neurons, layers)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=0)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Unscale the predictions
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Unscale the actual values
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_rescaled - predictions) ** 2))
    
    # Save predictions and actual values to CSV
    guess_data = pd.DataFrame({'ds': data['timestamp'].iloc[-len(y_test):].values, 'yhat': predictions.flatten()})
    eval_data = pd.DataFrame({'ds': data['timestamp'].iloc[-len(y_test):].values, 'y': y_test_rescaled.flatten()})
    guess_data.to_csv('guess_data.csv', index=False)
    eval_data.to_csv('eval_data.csv', index=False)
    
    # Save graphs
    # output_image_path = f"predictions_lstm_epochs_{epochs}_neurons_{neurons}_layers_{layers}.png"
    # plt.figure(figsize=(12, 6))
    # plt.plot(data['timestamp'].iloc[-len(y_test):], y_test_rescaled, label='Real value')
    # plt.plot(data['timestamp'].iloc[-len(y_test):], predictions, label='Predictions', alpha=0.7)
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.savefig(output_image_path)
    # plt.close()

    return rmse

# Example: Hyperparameter tuning
if __name__ == "__main__":
    # Define ranges for hyperparameters
    file_path = "data.csv"
    epochs_range = range(5, 31, 5)  # 5 to 30, step 5
    neurons_range = range(10, 51, 10)  # 10 to 50, step 10
    layers_range = range(1, 4)  # 1 to 3
    print("About to start!")
    results = []
    
    for epochs in epochs_range:
        for neurons in neurons_range:
            for layers in layers_range:
                print(f"Training with epochs={epochs}, neurons={neurons}, layers={layers}...")
                rmse = train_and_evaluate_lstm_model(file_path, epochs, neurons, layers)
                results.append([rmse, epochs, neurons, layers])
                print(f"RMSE: {rmse}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['RMSE', 'Epochs', 'Neurons', 'Layers'])
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    print("Results saved to hyperparameter_search_results.csv")
