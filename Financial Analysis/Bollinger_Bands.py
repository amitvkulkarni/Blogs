import pandas as pd
import numpy as np
import yfinance as yf
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt


tickers = ['AAPL']
start_date = '2020-01-01'
end_date = '2022-04-30'

# tickers = ['GOOG']
# start_date = '2018-01-01'
# end_date = '2023-04-30'


# Load the data from Yahoo finance
def load_data(tickers):
    try:            
        data = pd.DataFrame(columns = tickers)
        for ticker in tickers:
            data[ticker] = yf.download(ticker, start_date, end_date)['Close']
        
        return data
    except Exception as e:
        print(f'An exception occurred while executing load_data: {e}') 
        

df = load_data(tickers)
df.columns = ['Close']


# Creation of Bollinger Bands value and trends lines
sma = df.rolling(window=30).mean().dropna()
rstd = df.rolling(window=30).std().dropna()

upper_band = sma + 2 * rstd
lower_band = sma - 2 * rstd


upper_band = upper_band.rename(columns={'Close': 'upper'})
lower_band = lower_band.rename(columns={'Close': 'lower'})
df_bollinger_band = df.join(upper_band).join(lower_band)
df_bollinger_band = df_bollinger_band.dropna()


buyers = df_bollinger_band[df_bollinger_band['Close'] <= df_bollinger_band['lower']]
sellers = df_bollinger_band[df_bollinger_band['Close'] >= df_bollinger_band['upper']]


# Visualizing the Bolling Band
fig, ax = plt.subplots(figsize=(16,8))
plt.title(f'Bollinger Band - {tickers[0]}')
plt.ylabel('Price in USD')
plt.xlabel('Dates')
ax.plot(df_bollinger_band['Close'], label = 'Close Price', alpha = 0.25, color = 'black')
ax.plot(df_bollinger_band['upper'], label = 'Upper Band', alpha = 0.25, color = 'red')
ax.plot(df_bollinger_band['lower'], label = 'Lower Band', alpha = 0.25, color = 'blue')
ax.fill_between(df_bollinger_band.index, df_bollinger_band['upper'], df_bollinger_band['lower'], color = '#F9E79F')
# ax.fill_between(df_bollinger_band.index, df_bollinger_band['upper'], df_bollinger_band['lower'], color = '#DBE9FA')
ax.scatter(buyers.index, buyers['Close'], label = 'Buy', alpha = 1, marker = '^', color = 'green')
ax.scatter(sellers.index, sellers['Close'], label = 'Sell', alpha = 1, marker = 'v', color = 'red')
plt.legend()
plt.show()