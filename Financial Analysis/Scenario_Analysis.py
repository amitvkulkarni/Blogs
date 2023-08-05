######################################################################################
# Case Study 1: Scenario Analysis for Portfolio Risk Management
######################################################################################


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)["Adj Close"]


# Function to calculate daily returns
def calculate_returns(df):
    return df.pct_change().dropna()


# Function to perform scenario analysis
def scenario_analysis(stocks_data, num_scenarios):
    num_stocks = len(stocks_data.columns)
    num_days = len(stocks_data)

    # Initialize arrays to store results
    portfolio_returns = np.zeros(num_scenarios)
    portfolio_volatility = np.zeros(num_scenarios)
    portfolio_weights = np.zeros((num_scenarios, num_stocks))

    for i in range(num_scenarios):
        # Generate random weights
        weights = np.random.rand(num_stocks)
        weights /= np.sum(weights)  # Ensure weights sum up to 1

        # Calculate portfolio returns and volatility
        portfolio_returns[i] = np.sum(stocks_data.mean() * weights) * 252
        portfolio_volatility[i] = np.sqrt(
            np.dot(weights.T, np.dot(stocks_data.cov() * 252, weights))
        )
        portfolio_weights[i, :] = weights

    # Find the index of the portfolio with the highest Sharpe Ratio
    sharpe_ratios = portfolio_returns / portfolio_volatility
    max_sharpe_idx = np.argmax(sharpe_ratios)

    # Optimal portfolio weights for maximum Sharpe Ratio
    optimal_weights = portfolio_weights[max_sharpe_idx, :]

    return (
        portfolio_returns,
        portfolio_volatility,
        optimal_weights,
        max_sharpe_idx,
        sharpe_ratios,
    )


# Main code
if __name__ == "__main__":
    # Define the stocks and date range
    tickers = ["AAPL", "MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    # Fetch historical stock data
    stocks_data = get_stock_data(tickers, start_date, end_date)
    print(stocks_data)

    # Calculate daily returns
    returns_data = calculate_returns(stocks_data)
    print(returns_data)

    # Number of scenarios to create
    num_scenarios = 5000

    # Perform scenario analysis
    returns, volatility, weights, max_sharpe_idx, sharpe_ratios = scenario_analysis(
        returns_data, num_scenarios
    )

    # Print the results for the optimal portfolio
    print("Optimal Portfolio Weights:")
    print(list(zip(tickers, weights)))
    print("Expected Annual Return:", returns[max_sharpe_idx])
    print("Annual Volatility:", volatility[max_sharpe_idx])
    print("Sharpe Ratio:", sharpe_ratios[max_sharpe_idx])

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(volatility, returns, c=sharpe_ratios, cmap="viridis")
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Volatility (Annualized)")
    plt.ylabel("Expected Return (Annualized)")
    plt.title("Portfolio Optimization - Efficient Frontier")
    plt.scatter(
        volatility[max_sharpe_idx],
        returns[max_sharpe_idx],
        marker="*",
        color="r",
        s=200,
        label="Max Sharpe Ratio",
    )
    plt.legend()
    plt.grid(True)
    plt.show()


######################################################################################
# Case Study 3: Tradeoff between home loan tenure and interest rate
######################################################################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)

# Set up the basic variables and functions
principal = 7500000
interest_rates = [7, 8, 9, 10]
loan_terms = list(range(1, 21, 1))


def calculate_loan_payment(principal, interest_rate, loan_term):
    # Function to calculate the loan payment
    r = interest_rate / 100 / 12  # Monthly interest rate
    n = loan_term * 12  # Total number of months
    monthly_payment = (principal * r * (1 + r) ** n) / (((1 + r) ** n) - 1)
    return monthly_payment


# Create a DataFrame to store the results
results = []
for rate in interest_rates:
    for term in loan_terms:
        emi_amount = calculate_loan_payment(principal, rate, term)
        results.append([rate, term, emi_amount])

df = pd.DataFrame(
    results, columns=["Interest Rate (%)", "Loan Term (years)", "EMI Amount ($)"]
)

# Display the table
print(df)

# Create a heatmap to visualize the table
pivot_table = df.pivot("Interest Rate (%)", "Loan Term (years)", "EMI Amount ($)")
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_table,
    annot=False,
    fmt=".2f",
    cmap="flare",
    linewidths=1,
    cbar_kws={"label": "EMI Amount ($)"},
)
plt.title("EMI Amount for Different Interest Rates and Loan Terms")
plt.xlabel("Loan Term (years)")
plt.ylabel("Interest Rate (%)")
plt.show()
