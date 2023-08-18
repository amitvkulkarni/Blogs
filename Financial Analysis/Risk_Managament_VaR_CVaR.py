import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Define the stock symbol and time period
stock_symbol = "TSLA"
start_date = "2022-01-01"
end_date = "2023-01-01"

# Fetch historical stock price data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate daily returns
stock_data["Returns"] = stock_data["Adj Close"].pct_change()

# Calculate VaR and CVaR
confidence_level = 0.95
portfolio_value = 100000


returns = stock_data["Returns"].dropna()
sorted_returns = np.sort(returns)
var_index = int(np.floor(returns.shape[0] * (1 - confidence_level)))
var = np.percentile(returns, 5, interpolation="lower")
cvar = np.mean(sorted_returns[: var_index + 1])  # * initial_investment

print(f"The VaR = {round(var,4)}")
print(f"The CVaR = {round(cvar,4)}")

# Plot returns histogram with VaR and CVaR
plt.figure(figsize=(10, 6))

# Plot histogram
plt.hist(
    returns, bins=50, alpha=0.7, color="skyblue", edgecolor="skyblue", label="Returns"
)
sns.histplot(returns, color="skyblue", alpha=0.75, bins=50, kde=True)

# VaR and CVaR lines
plt.axvline(
    x=var,
    color="red",
    linestyle="dashed",
    linewidth=2,
    ymax=16,
    label=f"VaR ({confidence_level:.2f})",
)
plt.axvline(
    x=cvar,
    color="green",
    linestyle="dashed",
    linewidth=2,
    label=f"CVaR ({confidence_level:.2f})",
)
plt.axvline(
    x=returns.mean(),
    color="black",
    linestyle="dashed",
    linewidth=1,
    label=f"Mean",
)

left, bottom, width, height = (returns.mean(), 0, -(returns.mean() - var), 20)
rect = mpatches.Rectangle(
    (left, bottom),
    width,
    height,
    fill=True,
    color="red",
    alpha=0.2,
    facecolor="red",
    linewidth=2,
)
plt.gca().add_patch(rect)
left, bottom, width, height = (var, 0, -(var - cvar), 20)
# left, bottom, width, height = (returns.mean(), 0, -(returns.mean() - cvar), 16)
rect = mpatches.Rectangle(
    (left, bottom),
    width,
    height,
    fill=True,
    color="blue",
    alpha=0.2,
    facecolor="blue",
    linewidth=2,
)
plt.gca().add_patch(rect)
plt.text(
    (var + 0.002),
    10,
    f"  VaR {round(var, 4)} @ 5%",
    horizontalalignment="left",
    size="small",
    rotation=90,
    color="navy",
)
plt.text(
    cvar,
    12,
    f"CVaR {round(cvar, 4)} @ 5%  ",
    horizontalalignment="right",
    size="small",
    rotation=90,
    color="navy",
)


plt.title(f"Returns Distribution and VaR/CVaR for {stock_symbol}")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.legend()
plt.show()
