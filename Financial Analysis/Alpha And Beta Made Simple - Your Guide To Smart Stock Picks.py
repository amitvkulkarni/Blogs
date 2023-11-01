import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf


# Download historical stock price data using Yahoo Finance API
yf.pdr_override()
start_date = "2020-01-1"
end_date = "2022-01-1"
num_years = 2


stock_symbols = [
    "^GSPC",
    "AAPL",
    "GOOG",
    "MSFT",
    "TSLA",
    "HOG",
    "LUV",
    "T",
    "TXN",
    "KO",
    "TGT",
    "XOM",
    "WMT",
    "TXN",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
]

# stock_symbols = ["VMIAX","^GSPC"]
# stock_symbols = ["^GSPC"]
# stock_symbols = ["AAPL", "TSLA", "MSFT", "GOOG"]

# Calculate daily returns
returns = data["Adj Close"].pct_change()
market_returns = pdr.get_data_yahoo("^GSPC", start=start_date, end=end_date)[
    "Adj Close"
].pct_change()


####################################################################################
# Calculating expected returns for each of the stocks
####################################################################################
from scipy import stats

alpha_values = []
beta_values = []
ER = []
rf = 0


df_GSPC = pdr.get_data_yahoo("^GSPC", start=start_date, end=end_date)["Adj Close"]
total_returns = (df_GSPC[-1] - df_GSPC[0]) / df_GSPC[0]
rm = ((1 + total_returns) ** (1 / num_years)) - 1

for i, stock_symbol in enumerate(stock_symbols):
    # print(returns[stock_symbol].head(3))
    # print(stock_symbol)
    stock_returns = returns[stock_symbol].dropna()
    stock_beta, stock_alpha, _, _, _ = stats.linregress(
        market_returns.dropna(), stock_returns
    )
    alpha_values.append(stock_alpha)
    beta_values.append(stock_beta)
    ER_tmp = rf + (beta_values[i] * (rm - rf)) * 100
    ER.append(ER_tmp)
    print(f"Expected return for {stock_symbol} is  {round(ER_tmp,2)} %")


# Plot the risk and return chart
plt.figure(figsize=(12, 12))
plt.scatter(beta_values, ER, marker="o", color="b", label="Stocks")

# Add labels for each point
for i, symbol in enumerate(stock_symbols):
    plt.annotate(
        symbol,
        (beta_values[i], ER[i]),
        xytext=(10, -10),
        textcoords="offset points",
    )

plt.xlabel("Beta (Sensitivity)")
plt.ylabel("Expected Returns(%)")
plt.title("Risk-Return Plot")
plt.legend()
plt.grid(True)

alpha_threshold = round(rm * 100, 2)
beta_threshold = 1


# Add background colors for each quadrant
plt.axvline(x=beta_threshold, color="b", linestyle="--", lw=1)
plt.axhline(y=alpha_threshold, color="b", linestyle="--", lw=1)

# plt.show()

plt.fill_between(
    [0, beta_threshold],
    alpha_threshold,
    color="green",
    alpha=0.15,
)
plt.fill_between(
    [beta_threshold, 2],
    alpha_threshold,
    color="red",
    alpha=0.15,
)
plt.fill_between(
    [0, beta_threshold],
    alpha_threshold * 2,
    color="yellow",
    alpha=0.15,
)
plt.fill_between(
    [beta_threshold, 2],
    alpha_threshold * 2,
    color="blue",
    alpha=0.15,
)


plt.text(0.25, alpha_threshold * 1.75, "Low Risk High Returns", fontsize=12)
plt.text(1.5, alpha_threshold * 1.75, "High Risk High Returns", fontsize=12)
plt.text(0.25, 1.5, "Low Risk Low Return", fontsize=12)
plt.text(1.5, 1.5, "High RIsk Low Return", fontsize=12)
plt.xlabel("Beta")
plt.ylabel("Expected Returns(%)")
plt.title("Risk Vs Returns - Scatter Plot with Quadrants")
plt.grid(True)
plt.legend()

plt.show()
