import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1
import yfinance as yf
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch


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


def build_model(hp, X_train):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   kernel_regularizer=l1(hp.Choice('reg_strength', values=[0.0, 0.01, 0.1]))))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=False,
                   kernel_regularizer=l1(hp.Choice('reg_strength', values=[0.0, 0.01, 0.1]))))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
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

    tuner = RandomSearch(
        lambda hp: build_model(hp, X_train),
        objective='val_loss',
        max_trials=5,  # how many model variations to test
        executions_per_trial=3,  # how many trials per variation
        directory='my_dir',  # where to save the models
        project_name='helloworld')  # name of the project

    tuner.search_space_summary()

    tuner.search(X_train, Y_train,
                 epochs=20,
                 validation_data=(X_test, Y_test))

    best_model = tuner.get_best_models(num_models=1)[0]

    # Make predictions
    train_predict = best_model.predict(X_train)
    test_predict = best_model.predict(X_test)

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
