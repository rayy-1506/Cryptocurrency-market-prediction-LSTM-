import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load cryptocurrency data (e.g., Bitcoin data from Yahoo Finance)
def load_data(ticker='BTC-USD', start_date='2017-01-01', end_date='2024-01-01'):
data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']] # Using closing prices only
return data
# Feature engineering: Adding technical indicators
def add_technical_indicators(data):
# Adding Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands
data['BB_upper'] = data['SMA_50'] + 2 * data['Close'].rolling(window=50).std()
data['BB_lower'] = data['SMA_50'] - 2 * data['Close'].rolling(window=50).std()
# Drop NA values created by rolling calculations
data = data.dropna()
return data

# Prepare data for LSTM (Reshape data to 3D)
def create_dataset(data, time_step=60):
X, y = [], []
for i in range(len(data) - time_step - 1):
X.append(data[i:(i + time_step), :])
y.append(data[i + time_step, 0]) # Target is the 'Close' price
return np.array(X), np.array(y)

# Load data
data = load_data(ticker='BTC-USD', start_date='2017-01-01', end_date='2024-01-01')

# Add technical indicators
data = add_technical_indicators(data)
# Normalize data (Scale features between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# Create dataset with time step of 60 days
time_step = 60
X, y = create_dataset(scaled_data, time_step)
# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Build the LSTM model
model = Sequential()


# LSTM layers with Dropout for regularization
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],
X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
callbacks=[early_stop])
# Make predictions
predictions = model.predict(X_test)

# Inverse scale the predictions and actual values
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0],
scaled_data.shape[1] - 1)))))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1),
np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)))))[:, 0]
# Evaluate model performance
mse = mean_squared_error(y_test_actual, predictions)
print(f'Mean Squared Error (MSE): {mse}')


# Visualize results
plt.figure(figsize=(14,7))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
