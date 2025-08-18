#!/usr/bin/env python3
"""
LSTM Model Trainer for CryptoPulse

This script trains a Long Short-Term Memory (LSTM) model for time-series prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

class LSTMTrainer:
    def __init__(self, data_path, models_dir):
        self.data_path = data_path
        self.models_dir = models_dir
        self.model_dir = os.path.join(self.models_dir, 'lstm_model')
        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_prepare_data(self, look_back=10):
        """Load the dataset and prepare it for LSTM training."""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        # Use price and technical indicators as features
        features = df[['price_usd', 'price_ma_7', 'price_volatility']].copy()
        features.dropna(inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(scaled_features) - look_back):
            X.append(scaled_features[i:(i + look_back), :])
            y.append(scaled_features[i + look_back, 0]) # Predicting next day's price_usd

        return np.array(X), np.array(y), scaler

    def build_model(self, input_shape):
        """Build the LSTM model architecture."""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, look_back=10, epochs=50, batch_size=32):
        """Train the LSTM model."""
        X, y, scaler = self.load_and_prepare_data(look_back=look_back)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=1)

        model_path = os.path.join(self.model_dir, 'lstm_model.h5')
        model.save(model_path)

        scaler_path = os.path.join(self.model_dir, 'lstm_scaler.joblib')
        joblib.dump(scaler, scaler_path)

        print(f"LSTM model saved to {model_path}")
        return history

if __name__ == '__main__':
    DATA_PATH = "data/simplified_ml_dataset.csv"
    MODELS_DIR = "models"
    trainer = LSTMTrainer(DATA_PATH, MODELS_DIR)
    trainer.train()
