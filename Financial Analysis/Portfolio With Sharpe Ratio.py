##################################################################
# Import libraries
##################################################################

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import NormalDist
import statistics
import seaborn as sns
from datetime import date
from tabulate import tabulate
from nsepy import get_history as gh
from mplcursors import cursor 
plt.style.use('fivethirtyeight') #setting matplotlib style
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)


##################################################################
# Define configurations
##################################################################
initial_investment: int = 100000
start_date = date(2022,2,2)
# end_date = date.today()
end_date = date(2023,2,2)
# stocksymbols = ['TATAMOTORS','DABUR', 'ICICIBANK','WIPRO','INFY']
# weights = np.array([0.4, 0.2, 0.1, 0.1, 0.2])

stocksymbols = ['AXISBANK', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN']
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


##################################################################
# Data loading for given time period
##################################################################

def load_stock_data(start_date, end_date, investment: int, ticker: str):
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


df_stockPrice = load_stock_data(start_date, end_date, initial_investment, stocksymbols)
df_returns = df_stockPrice.pct_change().dropna()


df_returns['Portfolio'] = (weights * df_returns.values).sum(axis=1)
df_returns.head(3)

# Average daily return
df_returns['Portfolio'].mean()

# Standard deviation of the portfolio
df_returns['Portfolio'].std()
df_returns['Portfolio'].plot(kind='kde')


# Sharpe ratio
sharpe_ratio = df_returns['Portfolio'].mean() / df_returns['Portfolio'].std()



####################################################################################
# Histogram of stocks
####################################################################################

df_returns.hist(bins=100,figsize=(12,6))
plt.tight_layout()
plt.savefig('./Images/hist_banks.png', transparent=True)


####################################################################################
# Stock price trend
####################################################################################

fig, ax = plt.subplots(figsize=(15,8))
for i in df_stockPrice.columns.values :
    ax.plot(df_stockPrice[i], label = i)
ax.set_title("Portfolio Close Price History")
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Close Price INR (₨)' , fontsize=18)
ax.legend(df_stockPrice.columns.values , loc = 'upper right')
plt.show(fig)
# plt.savefig('./Images/price.png', transparent=True)

####################################################################################
# Stock price cumulative return trend
####################################################################################

daily_cummulative_simple_return =(df_returns+1).cumprod()
fig, ax = plt.subplots(figsize=(18,8))
for i in daily_cummulative_simple_return.columns.values :
    ax.plot(daily_cummulative_simple_return[i], lw =2 ,label = i)

ax.legend( loc = 'upper left' , fontsize =10)
ax.set_title('Daily Cumulative Simple returns/growth of investment')
ax.set_xlabel('Date')
ax.set_ylabel('Growth of ₨ 1 investment')
cursor(hover=True)
# plt.show()
# plt.savefig('./Images/Cumulative_trends.png', transparent=True)



####################################################################################
# Portfolio Simulation
####################################################################################
def sim_portfolio(weights):
    # port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    # return port_stdev
    port_mean = (weights * df_returns.values).sum(axis=1)
    # Annualized mean
    port_mean = port_mean * 252
    port_sd = np.sqrt(weights.T.dot(df_returns.cov() * 252 ).dot(weights))
    
    
    port_var = np.percentile(port_mean/252, 5, interpolation = 'lower')
    port_var_95.append(port_var)
    
    port_volatility.append(port_sd)
    mc_sim_sr = (port_mean.mean() / port_sd)
    
    return mc_sim_sr


port_var_95 = []
port_returns = []
port_volatility = []
port_weights = []
sharpe_ratio = []

num_assets = len(stocksymbols)
num_portfolios = 25000
np.random.seed(1357)
for port in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/sum(weights)
    port_weights.append(weights)
    
    df_wts_returns = df_returns.mean().dot(weights)
    port_returns.append(df_wts_returns*100)
    
    mc_sim_sr = sim_portfolio(weights)
    sharpe_ratio.append(mc_sim_sr)
    

port_weights = [[round(wt[0],5), round(wt[1],5), round(wt[2],5),round(wt[3],5),round(wt[4],5)] for wt  in port_weights]
# dff = {'Returns': port_returns, 'Risk': port_volatility, 'Sharpe Ratio': sharpe_ratio, 'Weights': port_weights}
dff = {'Returns': port_returns, 'Risk': port_volatility, 'Sharpe Ratio': sharpe_ratio, 'Weights': port_weights, 'Port_var_95':port_var_95}
df_risk = pd.DataFrame(dff)
df_risk.tail(5)

# df_risk.iloc[df_risk['Port_var_95'].idxmax()]
# df_risk[df_risk['Port_var_95'] == df_risk['Port_var_95'].max()]
# df_risk[df_risk['Port_var_95'] == df_risk['Port_var_95'].min()]

####################################################################################
# Max and min sharpe ratio
####################################################################################
sr_max = df_risk[df_risk['Sharpe Ratio'] == df_risk['Sharpe Ratio'].max()]
sr_min = df_risk[df_risk['Sharpe Ratio'] == df_risk['Sharpe Ratio'].min()]



####################################################################################
# Efficient Frontier curve
####################################################################################

plt.figure(figsize=(12,8))
plt.scatter(port_volatility,
            port_returns,
            c=sharpe_ratio, 
            cmap='plasma')
plt.plot(sr_max['Risk'], sr_max['Returns'], marker = 'p', ms = 10, color = 'r')
plt.text(sr_max['Risk']+0.001, sr_max['Returns'], 'Max Sharpe Ratio')          
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# plt.savefig('./Images/Cumulative_trends.png', transparent=True)


####################################################################################
# Calculate and plot portfolio risk using VaR
####################################################################################

ddf_risk = df_returns.copy()
ddf_risk['Stock'] = (weights * ddf_risk.values).sum(axis=1)
ddf_risk.head(5)
ddf_risk['Stock'].max()

port_var_mean_95 = statistics.mean(port_var_95)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(ddf_risk['Stock'],color = 'blue',alpha = .2,bins = 50, 
                edgecolor = "black", kde = True)
plt.xlabel('Volatility') 
plt.ylabel('Stock Return')
plt.text(port_var_mean_95, 23, f'VAR {round(port_var_mean_95, 4)} @ 5%', 
         horizontalalignment='right', 
         size='small', color='navy')
left, bottom, width, height = (port_var_mean_95, 0, -5, 1750)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=True,
                        color="red",
                        alpha=0.2,
                        facecolor="red",
                       linewidth=2)
plt.gca().add_patch(rect)







