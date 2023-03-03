import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nsepy import get_history as gh
from datetime import date

####################################################################################
# Initial Configurations
####################################################################################
start_date = date(2018,2,2)
end_date = date(2023,2,2)
tickers = ['TATAMOTORS','DABUR', 'ICICIBANK','WIPRO','INFY']


####################################################################################
# Function To Load The Data
####################################################################################
def load_stock_data(start_date, end_date, ticker):
    df = pd.DataFrame()
    for i in range(len(ticker)):
        data = gh(symbol=ticker[i],start= start_date, end=end_date)[['Symbol','Close']]
        data.rename(columns={'Close':data['Symbol'][0]},inplace=True)
        data.drop(['Symbol'], axis=1,inplace=True)
        if i == 0:
            df = data
        if i != 0:
            df = df.join(data)
    
    return df

# Loading the stock data in the tickers from NSEPY data for specified period
df_stock = load_stock_data(start_date, end_date, tickers)

# Loading the NIFTY data for specified period
df_nifty = gh(symbol = 'NIFTY',start = date(2018,2,2), end = date(2023,2,2), index= True)[['Close']]
df_nifty.rename(columns = {'Close':'NIFTY'}, inplace = True)
df_port = pd.concat([df_stock, df_nifty], axis = 1)

# Calculating daily returns
df_returns = df_port.pct_change().dropna().reset_index()


####################################################################################
# Exploratory Data Analysis - EDA
####################################################################################

####################################################################################
# Daily Closing Price Trend
####################################################################################

plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("Daily closing prices", fontsize=18, y=0.95)
for n, ticker in enumerate(tickers):
    ax = plt.subplot(3, 2, n + 1)
    df_stock[ticker].plot(ax= ax)
    ax.set_title(ticker.upper())
    ax.set_xlabel("")
plt.savefig('./Images/price.png', transparent=True)
    
    
####################################################################################
# Daily Returns Trend
####################################################################################

plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("Daily Returns", fontsize=18, y=0.95)
for n, ticker in enumerate(tickers):
    ax = plt.subplot(3, 2, n + 1)    
    df_returns[ticker].plot(ax= ax)       
    ax.set_title(ticker.upper())
    ax.set_xlabel("")
plt.savefig('./Images/returns.png', transparent=True)



####################################################################################
# Fitting a regression line for each of the stock
####################################################################################
beta = []
alpha = []

for i in df_returns.columns:
  if i != 'Date' and i != 'NIFTY':
    df_returns.plot(kind = 'scatter', x = 'NIFTY', y = i)
    b, a = np.polyfit(df_returns['NIFTY'], df_returns[i], 1)
    plt.plot(df_returns['NIFTY'], b * df_returns['NIFTY'] + a, '-', color = 'r')  
    beta.append(b)    
    alpha.append(a) 

    
####################################################################################    
# Calculating expected returns for each of the stocks
####################################################################################
ER = []
rf = 0 
rm = df_returns['NIFTY'].mean() * 252
print(f'Market return is {rm}%')
for i, b in enumerate(beta):
    ER_tmp = rf + (b * (rf - rm)) * 100 
    ER.append(ER_tmp)
    print(f'Expected return based on CAPM for {tickers[i]} is  {round(ER_tmp,2)} %')
  
####################################################################################
# Assigning weights for the stocks
####################################################################################
portfolio_weights = 1/len(tickers) * np.ones(len(tickers)) 
ER_portfolio = sum(ER * portfolio_weights)
print(f'Portfolio expected return is {round(ER_portfolio,2)} %')



####################################################################################
# SML
####################################################################################
sml_beta = pd.DataFrame.from_dict(beta, orient = 'index', columns=['beta'])
sml_returns = pd.DataFrame.from_dict(ER, orient = 'index', columns=['ER'])
sml = pd.concat([sml_beta, sml_returns], axis = 1)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(sml['beta'], sml['ER'])

left, bottom, width, height = (0.6, 0.06, 0.4, 0.06)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=True,
                        color="red",
                        alpha=0.1,
                        facecolor="red",
                       linewidth=2)

ax.set_title('Security Market Line')
ax.set_xlabel('Beta')
ax.set_ylabel('Expected Returns')
plt.gca().add_patch(rect)



####################################################################################
# Regression Line For Each Of The Stocks
####################################################################################


plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("", fontsize=18, y=0.95)
for n, ticker  in enumerate(tickers):
    ax = plt.subplot(3, 2, n + 1)      
    plt.plot(df_returns['NIFTY'], df_returns[ticker], 'o', alpha = 0.3)
    b, a = np.polyfit(df_returns['NIFTY'], df_returns[ticker], 1)
    plt.plot(df_returns['NIFTY'], b * df_returns['NIFTY'] + a, '-')  
    ax.set_title(ticker.upper())
    ax.set_xlabel('NIFTY')
    ax.set_ylabel(ticker)
    plt.tight_layout()
    plt.text(-0.03, 0.1, f'y = {round(a, 4)} + {round(b, 4)} * (rm - rf)', 
         horizontalalignment='right', 
         size='small', color='red')
plt.savefig('./Images/reg.png', transparent=True)

