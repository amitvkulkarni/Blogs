import pandas as pd
import numpy as np
import yfinance as yf
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


ticker = 'MSFT'
start_date = '2022-01-01'
end_date = '2023-01-01'

window_length = 14

# ticker = 'GOOG'
# start_date = '2016-01-01'
# end_date = '2023-04-30'

        
df = yf.Ticker(ticker).history(start= start_date, end=end_date)[['Open', 'High', 'Low', 'Close', 'Volume']]



df['diff'] = df['Close'].diff(1)

# Calculate Avg. Gains/Losses
df['gain'] = df['diff'].clip(lower=0).round(2)
df['loss'] = df['diff'].clip(upper=0).abs().round(2)

# df[['rs','rsi']].round(2)

# Creation of Bollinger Bands value and trends lines
sma = df.rolling(window=window_length).mean().dropna()
rstd = df.rolling(window=window_length).std().dropna()

upper_band = sma + 2 * rstd
lower_band = sma - 2 * rstd

upper_band = upper_band.rename(columns={'Close': 'upper'})['upper']
lower_band = lower_band.rename(columns={'Close': 'lower'})['lower']
df_bollinger_band = df.join(upper_band).join(lower_band)
df_bollinger_band = df_bollinger_band.dropna()

df_bollinger_band[['upper', 'lower']].round(2)

# Get initial Averages
df['avg_gain'] = df['gain'].rolling(window_length).mean()
df['avg_loss'] = df['loss'].rolling(window_length).mean()

# Calculate RS Values
df['rs'] = df['avg_gain'] / df['avg_loss']

# Calculate RSI
df['rsi'] = 100 - (100 / (1.0 + df['rs']))





def RSI(df):

        # Create Figure
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, row_width=[100, 250],
            specs=[[{"secondary_y": True}],[{"secondary_y": False}]]
        )


        # Create Candlestick chart for price data
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False    
        ),row=1, col=1, secondary_y=True,
        )
        fig.add_trace(go.Scatter(
            x= df_bollinger_band.index,
            y= df_bollinger_band['upper'],
            line=dict(color='red', width=1),
            # fill='tonexty',
            showlegend=False,
        ), row=1, col=1
        ),
        fig.add_trace(go.Scatter(
            x= df_bollinger_band.index,
            y= df_bollinger_band['lower'],
            line=dict(color='#DBE9FA', width=1),
            fill='tonexty',
            showlegend=False, 
        ), row=1, col=1
        ),
        fig.add_trace(go.Bar(
            x= df.index,
            y= df['Volume'],
            marker_color='Navy',
            showlegend=False,
        ), row=1, col=1,secondary_y=False,
        ),
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi'],
            line=dict(color='blue', width=1),
            showlegend=False,
        ), row=2, col=1
        )
        # Add upper/lower bounds
        fig.update_yaxes(range=[-10, 110], row=2, col=1)
        fig.add_hline(y=0, col=1, row=2, line_color="red", line_width=1)
        fig.add_hline(y=100, col=1, row=2, line_color="red", line_width=1)

        # Add overbought/oversold
        fig.add_hline(y=30, col=1, row=2, line_color='gray', line_width=1, line_dash='dot')
        fig.add_hline(y=70, col=1, row=2, line_color='gray', line_width=1, line_dash='dot')

        fig.update_layout( 
            title={
                'text': f'<b>Stock Price Movement with RSI for {ticker}</b>',
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                },
            font=dict(
                    family="Arial",
                    size=8,
                    color='IndianRed'
                ),
            plot_bgcolor ='white'
            
                        
        ),

        fig.update_yaxes(title_text="<b>Stock Prices</b>", secondary_y=False, row=1, col=1, title_font=dict(size=8),tickfont=dict(size = 8))
        fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, row=1, col=1, title_font=dict(size=8),tickfont=dict(size = 8))
        fig.update_xaxes(row=1, col=1,tickfont=dict(size = 8))

        fig.update_yaxes(title_text="<b>RSI</b>", secondary_y=False, row=2, col=1, title_font=dict(size=8),tickfont=dict(size = 8))
        fig.update_xaxes(title_text="<b>Period</b>", row=2, col=1, title_font=dict(size=8),tickfont=dict(size = 8))


        fig.add_shape(                
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=-0.1,
                    y0=-0.2,
                    x1=1.02,
                    y1=1.0,
                    line=dict(
                        color="gray",
                        width=1.5,
                    )
                )
        fig.update_xaxes(rangeslider_visible=False)
        fig.layout.yaxis2.showgrid=True
        fig.show()
        


RSI(df)








