##################################################################
# Import libraries
##################################################################

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import date
from tabulate import tabulate
from nsepy import get_history as gh
plt.style.use('fivethirtyeight') #setting matplotlib style
warnings.filterwarnings('ignore')


class LoadData:   
        
    ##################################################################
    # Define configurations
    ##################################################################
    initial_investment = 100000
    startdate = date(2022,2,2)
    # end_date = date.today()
    end_date = date(2023,2,2)
    stocksymbols = ['TATAMOTORS']

    ##################################################################
    # Load data for a stock from nsepy
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
    
    df_stock_data = load_stock_data(startdate, end_date, initial_investment, stocksymbols)

    ##################################################################
    # Calculate the simple return
    ##################################################################
   
    df_returns = df_stock_data.copy()
    df_returns.columns = ['Stock']
    df_returns = df_returns.pct_change()
    df_returns.dropna(inplace=True)    
    
    
# Create an instance of the data load class
obj_loadData = LoadData()


class StockRisk:      
    
    def __init__(self, start_date, end_date, investment: int, ticker: str):
        """Initialization

        Args:
            start_date (Date): Indicates the date from which the stock data will be extracted
            end_date (Date): Indicates the date till which the stock data will be extracted. In this case it is today
            investment (int): The initial investment value
            ticker (str): The name of the stock
        """
        self.start_date = start_date
        self.end_date = end_date
        self.investment = investment
        self.ticker = ticker
        
      
    def var_historical(self):
        
        # returns = df_returns.copy()
        returns = obj_loadData.df_returns.copy()
        # check for variance
        var_stock = returns.var()

        # Calculate mean returns of the stock
        avg_rets = returns.mean()        

        # Calculate SD of the  stock
        avg_std = returns.std()        
        

        returns['Stock'].mean()
        returns['Stock'].median()
        returns['Stock'].quantile(.05, 'lower')
        # var_hist = returns['Stock'].quantile(.05, 'lower')
        # var_hist = var_hist
        returns.sort_values('Stock').head(13)
        var_hist = np.percentile(returns['Stock'], 5, interpolation = 'lower')     
       
        print(tabulate([[self.ticker,avg_rets,avg_std,var_hist]],
                       headers = ['Mean', 'Standard Deviation', 'VaR %'],
                       tablefmt = 'fancy_grid',stralign='center',numalign='center',floatfmt=".4f"))
        print(var_hist)
        return var_hist

    def var_bootstrap(self,iterations: int):
        """Bootstrap

        Args:
            iterations (int): The number of times the resampling is carried out 
        """
                
        def var_boot(data):            
            # return np.percentile(data, 5, interpolation=”lower”)
            # print(f'The data is :{data}')
            dff = pd.DataFrame(data, columns = ['sample'])
            return np.percentile(dff, 5, interpolation = 'lower')
            # return dff['sample'].quantile(.05, 'lower')
                
        def bootstrap(data, func):
            sample = np.random.choice(data, len(data))
            return func(sample)


        def generate_sample_data(data, func, size):
            bs_replicates = np.empty(size)
            for i in range(size):
                bs_replicates[i] = bootstrap(data, func)
            return bs_replicates

        # returns = self.stock_returns()
        returns = obj_loadData.df_returns.copy()
        

        bootstrap_VaR = generate_sample_data(returns['Stock'], var_boot, iterations)
        var_bootstrap = np.mean(bootstrap_VaR)
        print(f'The Bootstrap VaR measure is {np.mean(bootstrap_VaR)}')
        return np.mean(bootstrap_VaR)


    def var_weighted_decay_factor(self):
        
        # returns = self.stock_returns()
        returns = obj_loadData.df_returns.copy()
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
        
    def show_summary(self):
        
        var_hist = self.var_historical()
        var_bs = self.var_bootstrap(500)
        var_decay = self.var_weighted_decay_factor()
        
        print(tabulate([[self.ticker,var_hist,var_bs,var_decay]],
                       headers = ['Historical', 'Bootstrap', 'Decay'],
                       tablefmt = 'fancy_grid',stralign='center',numalign='center',floatfmt=".4f"))
        

    
    def plot_stock_price(self):
        # stock_price = self.load_stock_data()
        stock_price = obj_loadData.df_stock_data.copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = sns.histplot(stock_price,color = 'blue',alpha = .2, bins = 50, kde = True)
        return fig
    
    
    def plot_stock_returns(self):
        # returns = self.stock_returns()
        returns = obj_loadData.df_returns.copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = sns.histplot(returns,color = 'blue',alpha = .2, bins = 50, kde = True)
        return fig
    

    def plot_shade(self, var_returns):
        # returns = self.stock_returns()
        returns = obj_loadData.df_returns.copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(returns,color = 'blue',alpha = .2,bins = 50, kde = True)
        plt.axvline(var_returns, 0, 25, color = 'red', alpha = .15)
        plt.text(var_returns, 25, f'VAR {round(var_returns, 4)} @ 5%', 
                 horizontalalignment='right', 
                 size='small', 
                 color='navy')
        left, bottom, width, height = (var_returns, 0, -5, 30)
        rect=mpatches.Rectangle((left,bottom),width,height, 
                                fill=True,
                                color="red",
                                alpha=0.2,
                                facecolor="red",
                                linewidth=2)
        plt.gca().add_patch(rect)
        
        
    
    def plot_stock_returns_shaded(self, var_method):

        if var_method == 'historical':
            self.plot_shade(self.var_historical())    
                
        elif var_method == 'bootstrap':
            self.plot_shade(self.var_bootstrap(500))
        else:
            self.plot_shade(self.var_weighted_decay_factor())


#################################################################################################
# Execution
#################################################################################################

obj_var = StockRisk(startdate, end_date, initial_investment, stocksymbols)

obj_var.var_historical()
obj_var.var_bootstrap(500)
obj_var.var_weighted_decay_factor()
obj_var.show_summary()

obj_var.plot_stock_returns_shaded('historical')
obj_var.plot_stock_returns_shaded('bootstrap')
obj_var.plot_stock_returns_shaded('decay')

