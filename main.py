import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1
import yfinance as yf
import matplotlib.pyplot as plt

def load_data(symbol):
    data = yf.download(symbol, start='2000-01-01')
    close_data = data['Close'].values.reshape(-1, 1)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']], close_data

def preprocess_data(data, close_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = close_scaler.fit_transform(close_data)
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size, :], data[train_size:len(data), :]
    close_train, close_test = close_data[0:train_size, :], close_data[train_size:len(close_data), :]
    return train, test, scaler, close_train, close_test, close_scaler


def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 3])  # 'Close' price is the target
    return np.array(X), np.array(Y)

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l1(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l1(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    data, close_data = load_data('AAPL')
    train, test, scaler, close_train, close_test, close_scaler = preprocess_data(data, close_data)
    X_train, Y_train = create_dataset(train)
    X_test, Y_test = create_dataset(test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    model = create_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions back to original scale
    train_predict = close_scaler.inverse_transform(train_predict)
    Y_train = close_scaler.inverse_transform([Y_train])
    test_predict = close_scaler.inverse_transform(test_predict)
    Y_test = close_scaler.inverse_transform([Y_test])

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test[0], label='Actual')
    plt.plot(test_predict[:, 0], label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

