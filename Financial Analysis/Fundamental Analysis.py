import numpy as np
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si



class FundamentalRatio:
    
    def __init__(self, start_date, end_date, tickers, ratio_stat):
        
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.ratio_stat = ratio_stat
    
    
    def load_data(self):
        try:            
            data = pd.DataFrame(columns = self.tickers)
            for ticker in self.tickers:
                data[ticker] = yf.download(self.ticker, 
                                        self.start_date,
                                        self.end_date)['Adj Close']
            
            return data
        except Exception as e:
            print(f'An exception occurred while executing load_data: {e}')        
    

    def get_fundamental(self):
        
        try:
            df_fundamentals = pd.DataFrame()
            
            for ticker in range(len(self.tickers)):
                stock_name = str(self.tickers[ticker])
                fundamental_ratio = si.get_stats_valuation(stock_name)
                fundamental_ratio.index = fundamental_ratio[0]
                fundamental_ratio = fundamental_ratio.drop(labels=0,axis=1)
                tmp_table = fundamental_ratio.T
                tmp_table = tmp_table[self.ratio_stat]
                df_fundamentals = df_fundamentals.append(tmp_table)

            df_fundamentals.index = self.tickers
            df_fundamentals = df_fundamentals.astype('float')
            # df_fundamentals = pd.to_numeric(df_fundamentals['Trailing P/E'])
            df_fundamentals.dropna(inplace = True)
            return df_fundamentals
        except Exception as e:
            print(f'An exception occurred while executing get_fundamental: {e}')        


    def get_over_under_stocks(self):
        
        try:        
            df_fundamentals = self.get_fundamental()
            df_fundamentals['Trailing P/E'].mean()
            df_fundamentals['over_under'] = (df_fundamentals['Trailing P/E']) / (df_fundamentals['Trailing P/E'].mean())
           
            category = []
            for i in df_fundamentals['over_under']:
                if i < 1: category.append('Under Valued')
                elif i>1: category.append('Over Valued')
                else: category.append('Fair Valued')
            
            df_fundamentals['Category'] = category
            return df_fundamentals
        except Exception as e:
            print(f'An exception occurred while executing get_fundamental: {e}')   
    

##################################################################
# Initiate execution
##################################################################
start_date = '2022-02-02'
end_date = '2023-02-02'

tickers = ['AAPL', 'IBM', 'MSFT', 'WMT', 'AMGN', 'AXP', 'BA','NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']

ratio_stat = ['Trailing P/E','Forward P/E','PEG Ratio (5 yr expected)', 'Price/Book (mrq)',
              'Price/Sales (ttm)','Enterprise Value/EBITDA', 'Enterprise Value/Revenue']

obj_fundamentals = FundamentalRatio(start_date, end_date, tickers, ratio_stat)
# obj_fundamentals.get_fundamental()

df_classify = obj_fundamentals.get_over_under_stocks()
df_classify

# Filter the under valued stocks
df_classify[df_classify['Category'] == 'Under Valued']
