import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Step 1: Data Collection
stock_symbol = 'AAPL'  # Example stock symbol (Apple Inc.)
data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Step 2: Data Preprocessing
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())

# Step 3: Feature Engineering
features = ['Date_ordinal']
target = 'Close'

X = data[features]
y = data[target]

# Step 4: Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Step 6: Visualization
plt.figure(figsize=(14, 7))

plt.plot(data['Date'], data['Close'], label='Actual Price')
plt.plot(data['Date'].iloc[X_train.index], y_pred_train, label='Train Predictions', linestyle='--')
plt.plot(data['Date'].iloc[X_test.index], y_pred_test, label='Test Predictions', linestyle='--')

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
