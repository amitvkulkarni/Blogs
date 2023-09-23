##################################################################
# Import the libraries
##################################################################
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


##################################################################
# Define the time period and the stock
##################################################################
# Define the stock symbol and date range
stock_symbol = "AAPL"
start_date = "2022-01-01"
end_date = "2023-01-01"

# Download historical stock data
df = yf.download(stock_symbol, start=start_date, end=end_date)


##################################################################
# Explore the data, the data types and statistics
##################################################################
df.head()
df.describe()

##################################################################
# Price trends of AAPL
##################################################################
fig, ax = plt.subplots(figsize=(16, 8))
plt.title(f"Stock Trend Over The Years - {stock_symbol}")
plt.ylabel("Price in USD")
plt.xlabel("Period")
ax.plot(df["Close"], label="Close Price", alpha=0.9, color="blue")


##################################################################
# Calculate RSI (Relative Strength Index)
##################################################################
def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


df["RSI"] = calculate_rsi(df)


##################################################################
# Calculate MACD (Moving Average Convergence Divergence)
##################################################################
def calculate_macd(data, short_window=12, long_window=26):
    ema_short = data["Close"].ewm(span=short_window, min_periods=1, adjust=False).mean()
    ema_long = data["Close"].ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd = ema_short - ema_long
    signal_line = macd.ewm(span=9, min_periods=1, adjust=False).mean()

    return macd, signal_line


df["MACD"], df["Signal_Line"] = calculate_macd(df)


##################################################################
# Calculate Simple Moving Average (SMA)
##################################################################
def calculate_sma(data, window=50):
    sma = data["Close"].rolling(window=window, min_periods=1).mean()
    return sma


df["SMA"] = calculate_sma(df)


##################################################################
# Calculate Exponential Moving Average (EMA)
##################################################################
def calculate_ema(data, window=12):
    ema = data["Close"].ewm(span=window, min_periods=1, adjust=False).mean()
    return ema


df["EMA"] = calculate_ema(df)


##################################################################
# Create a binary target variable (1 for Buy, 0 for Hold/Sell)
##################################################################
df["Signal"] = np.where(
    (df["RSI"] > 30)
    & (df["MACD"] > df["Signal_Line"])
    & (df["Close"] > df["SMA"])
    & (df["Close"] > df["EMA"]),
    1,
    0,
)

df.dropna(inplace=True)


##################################################################
# Prepare the features (RSI, MACD, SMA, and EMA) and target (Signal)
# for the machine learning model
##################################################################

X = df[["RSI", "MACD", "SMA", "EMA"]].values
y = df["Signal"].values


###################################################################################
# Function for model building, model's performance measurement and visualization
###################################################################################


def build_model():
    # Split the data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = (
        X[:train_size],
        X[train_size:],
        y[:train_size],
        y[train_size:],
    )

    # Create and train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Generate predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate ROC curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 8})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Backtesting the trading strategy
    df["Strategy_Return"] = df["Close"].pct_change() * df["Signal"].shift(1)
    df["Buy_Hold_Return"] = df["Close"].pct_change()

    # Calculate cumulative returns
    df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()
    df["Cumulative_Buy_Hold_Return"] = (1 + df["Buy_Hold_Return"]).cumprod()

    # Plot the stock's closing price, RSI, MACD, SMA, and EMA
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(df["Close"], label="Close Price", color="blue")
    plt.title(f"{stock_symbol} Stock Price")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df["RSI"], label="RSI", color="orange")
    plt.axhline(30, color="red", linestyle="--", alpha=0.5, label="RSI Oversold (30)")
    plt.axhline(
        70, color="green", linestyle="--", alpha=0.5, label="RSI Overbought (70)"
    )
    plt.title("Relative Strength Index (RSI)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df["MACD"], label="MACD", color="purple")
    plt.plot(df["Signal_Line"], label="Signal Line", color="yellow")
    plt.title("MACD and Signal Line")
    plt.legend()
    plt.tight_layout(h_pad=5, w_pad=5)

    # Plot the buy/sell signals generated by the machine learning model
    plt.figure(figsize=(12, 4))
    plt.plot(
        df.index[train_size:],
        y_pred,
        label="ML Model Signal",
        color="green",
        marker="o",
    )
    plt.title(f"{stock_symbol} Buy/Sell Signal (ML Model)")
    plt.xlabel("Date")
    plt.ylabel("Signal (1=Buy, 0=Hold/Sell)")
    plt.legend()

    # Plot ROC curve
    plt.figure(figsize=(4, 3))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Plot cumulative returns of the strategy and buy-and-hold
    plt.figure(figsize=(12, 6))
    plt.plot(
        df.index,
        df["Cumulative_Strategy_Return"],
        label="Strategy Return",
        color="green",
    )
    plt.plot(
        df.index,
        df["Cumulative_Buy_Hold_Return"],
        label="Buy and Hold Return",
        color="blue",
    )
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()

    plt.figure(figsize=(4, 2))
    coef_values = list(abs(model.coef_[0]))
    feature_names = ["RSI", "MACD", "SMA", "EMA"]  # Update with your feature names
    plt.barh(feature_names, sorted(coef_values, reverse=False), color="purple")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Variable Importance (Coefficients)")
    plt.show()

    # Display accuracy
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    plt.show()


##################################################################
# Call the function for execution
##################################################################
build_model()
