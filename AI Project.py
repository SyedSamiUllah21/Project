import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch stock data (Example: Tesla stock)
stock_symbol = 'TSLA'
stock_data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# View the first few rows of the dataset
print(stock_data.head())

# Use the 'Adj Close' column for prediction
stock_data = stock_data[['Adj Close']]

# Create a column to represent future prices (shifted by 1 day)
stock_data['Prediction'] = stock_data['Adj Close'].shift(-1)

# Drop the last row (which has NaN for 'Prediction' due to the shift)
stock_data.dropna(inplace=True)

# Define features (X) and target (y)
X = np.array(stock_data[['Adj Close']])
y = np.array(stock_data['Prediction'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model and calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict future prices using the test data
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.title(f'Stock Price Prediction for {stock_symbol}')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
