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
stocksymbols = ['TATAMOTORS','DABUR', 'ICICIBANK','WIPRO','INFY']
weights = np.array([0.4, 0.2, 0.1, 0.1, 0.2])


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


portfolio = (weights * df_returns.values).sum(axis=1)
port_var = np.percentile(portfolio, 5, interpolation = 'lower')
port_var


##################################################################
# Bootstrap simulation
##################################################################

def sim_portfolio(weights):
    # port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    # return port_stdev
    tmp_pp = (weights * df_returns.values).sum(axis=1)
    var_sim_port = np.percentile(tmp_pp, 5, interpolation = 'lower')
    return var_sim_port


port_returns = []
port_volatility = []
port_weights = []

num_assets = len(stocksymbols)
num_portfolios = 10000
np.random.seed(1357)
for port in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/sum(weights)
    port_weights.append(weights)
    
    df_wts_returns = df_returns.mean().dot(weights)
    port_returns.append(df_wts_returns*100)
    
    var_port_95 = sim_portfolio(weights)
    port_volatility.append(var_port_95)
    

port_weights = [[round(wt[0],5), round(wt[1],5), round(wt[2],5),round(wt[3],5),round(wt[4],5)] for wt  in port_weights]
dff = {'Returns': port_returns, 'Risk': port_volatility, 'Weights': port_weights}
df_risk = pd.DataFrame(dff)
df_risk['Risk'].mean()

df_risk[df_risk['Risk']]

df_risk[(df_risk['Risk'] > -0.02) & (df_risk['Returns'] > 0.01)]


df_risk['Risk'].max()
df_risk['Risk'].min()

df_risk['Returns'].max()
df_risk['Returns'].min()


##################################################################
# Selecting new weights for max risk
##################################################################
max_risk = df_risk.iloc[df_risk['Risk'].idxmin()]
max_risk[0]
max_risk[1]
max_risk[2]


min_risk = df_risk.iloc[df_risk['Risk'].idxmax()]
min_risk[0]
min_risk[1]
min_risk[2]

max_returns = df_risk.iloc[df_risk['Returns'].idxmax()]
max_returns[0]
max_returns[1]
max_returns[2]

min_returns = df_risk.iloc[df_risk['Returns'].idxmin()]
min_returns[0]
min_returns[1]
min_returns[2]

port_var_basic_sim = sim_portfolio(min_risk[2])
port_var_basic_sim



##################################################################
# Selecting new weights for max risk
##################################################################

def var_weighted_decay_factor(weights):
    
    returns = df_returns.copy()
    returns['Stock'] = (weights * returns.values).sum(axis=1)
    # returns = returns['Stock']
    decay_factor = 0.5 #we’re picking this arbitrarily
    n = len(returns)
    wts = [(decay_factor**(i-1) * (1-decay_factor))/(1-decay_factor**n) for i in range(1, n+1)]

    #Need to reverse the PnL to put recent returns on top
    returns_recent_first = returns[::-1]
    weights_dict = {'Returns':returns_recent_first, 'Weights':wts}

    wts_returns = pd.DataFrame(returns_recent_first['Stock'])
    wts_returns['wts'] = wts

    sort_wts = wts_returns.sort_values(by='Stock')
    sort_wts['Cumulative'] = sort_wts.wts.cumsum()
    sort_wts

    #Find where cumulative (percentile) hits 0.05
    sort_wts = sort_wts.reset_index()
    idx = sort_wts[sort_wts.Cumulative <= 0.05].Stock.idxmax()
    sort_wts.filter(items = [idx], axis = 0)


    xp = sort_wts.loc[idx:idx+1, 'Cumulative'].values
    fp = sort_wts.loc[idx:idx+1, 'Stock'].values
    var_decay = np.interp(0.05, xp, fp) 

    print(f'The VAR of stock using decay factor is {var_decay} ')
    return var_decay


port_var_decay = var_weighted_decay_factor(weights)

##################################################################
# Pie chart of portfolio
##################################################################

def plot_portflio_allocation(weights):

    fig, ax = plt.subplots(figsize=(5, 5))
    # plt.pie(max_returns[2], labels= stocksymbols)

    # Creating explode data
    explode = (0.1, 0.1, 0.1, 0.1, 0.1)
    
    # Creating color parameters
    colors = ( "orange", "green", "brown",
            "grey", "indigo", "beige")
    
    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }
    
    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))        
        return f'{round(pct,2)} %'
    
    
    # Creating plot
    # fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(weights,
                                    autopct = lambda pct: func(pct, weights),
                                    explode = explode,
                                    labels = stocksymbols,
                                    shadow = True,
                                    colors = colors,
                                    startangle = 90,
                                    wedgeprops = wp,
                                    textprops = dict(color ="w"))
    
    # Adding legend
    ax.legend(wedges, stocksymbols,
            title ="Portfolio",
            loc ="lower right",
            bbox_to_anchor =(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Financial Portfolio")
    
    return fig


plot_portflio_allocation(max_risk[2])



##################################################################
# Plot for stock price trends
##################################################################

def plot_price_trends():
    fig, ax = plt.subplots(figsize=(15,8))
    for i in df_stockPrice.columns.values :
        ax.plot(df_stockPrice[i], label = i)
    ax.set_title("Portfolio Close Price History")
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price INR (₨)' , fontsize=18)
    ax.legend(df_stockPrice.columns.values , loc = 'upper right')
    plt.show(fig)


plot_price_trends()

##################################################################
# Plot for daily return trend
##################################################################
def plot_daily_return_trends():

    print('Daily simple returns')
    fig, ax = plt.subplots(figsize=(15,8))
    for i in df_returns.columns.values :
        ax.plot(df_returns[i], lw =2 ,label = i)

    ax.legend( loc = 'upper right' , fontsize =10)
    ax.set_title('Volatility in Daily simple returns ')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily simple returns')
    plt.show(fig)

    print('Average Daily returns(%) of stocks in your portfolio')
    Avg_daily = df_returns.mean()
    print(Avg_daily*100)

    print('Annualized Standard Deviation (Volatality(%), 252 trading days) of individual stocks in your portfolio on the basis of daily simple returns.')
    print(df_returns.std() * np.sqrt(252) * 100)

    print('Return per unit of risk')
    Avg_daily / (df_returns.std() * np.sqrt(252)) *100


plot_daily_return_trends()    


##################################################################
# Plot for cumulative trend
##################################################################

def plot_daily_cumulative_returns():

    daily_cummulative_simple_return =(df_returns+1).cumprod()
    daily_cummulative_simple_return
    #visualize the daily cummulative simple return
    print('Cummulative Returns')
    fig, ax = plt.subplots(figsize=(18,8))

    for i in daily_cummulative_simple_return.columns.values :
        ax.plot(daily_cummulative_simple_return[i], lw =2 ,label = i)

    ax.legend( loc = 'upper left' , fontsize =10)
    ax.set_title('Daily Cumulative Simple returns/growth of investment')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of ₨ 1 investment')
    cursor(hover=True)
    plt.show()


plot_daily_cumulative_returns()

##################################################################


# weights = [0.05294, 0.34003, 0.30989, 0.18632, 0.11082]
ddf = df_returns.copy()
ddf['Stock'] = (weights * ddf.values).sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(False)
sns.histplot(ddf['Stock'],color = 'blue',alpha = .2,bins = 50, 
                edgecolor = "black", kde = True)
plt.text(port_var_basic_sim, 17, f'VAR {round(port_var_basic_sim, 4)} @ 5%', 
         horizontalalignment='right', 
         size='small', color='navy')
left, bottom, width, height = (port_var_basic_sim, 0, -5, 550)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=True,
                        color="red",
                        alpha=0.2,
                        facecolor="red",
                       linewidth=2)

ax.set_title('Distribution of simulated returns')
ax.set_xlabel('Weighed stock returns')
ax.set_ylabel('Frequency')

plt.gca().add_patch(rect)