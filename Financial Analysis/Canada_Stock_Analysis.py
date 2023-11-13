import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


######################################################################
# Load Data
######################################################################
def get_data(start_date=None, end_date=None):
    try:
        df_stock_price = pd.read_csv("./top_50_canadian_stocks_data_since_2010.csv")
        if start_date is not None and end_date is not None:
            df_stock_price = df_stock_price[
                (df_stock_price["Date"] >= start_date)
                & (df_stock_price["Date"] <= end_date)
            ]
            return df_stock_price
        else:
            return df_stock_price
    except Exception as e:
        print(f"Error found executing get_data(): {e}")


######################################################################
# Basic EDA
######################################################################
def get_basic_stats(df):
    print("-------------HEAD------------------------")
    print(df.head())
    print("-------------INFO------------------------")
    print(df.info())
    print("-------------DESCRIBE--------------------")
    print(df.describe())
    print("-------------FInd NULL-------------------")
    print(df.isnull().sum())
    print("-------------IDENTIFY NULL COLUMNS-------")
    print(df.columns[df.isna().any()].tolist())


######################################################################
# Data Cleaning
######################################################################


def get_clean_data(df):
    try:
        remove_Stocks = df.columns[df.isna().any()].tolist()
        df_clean = df.drop(remove_Stocks, axis=1)
        return df_clean
    except Exception as e:
        print(f"Error found executing get_clean_data(): {e}")


######################################################################
# Stock and industry mapping
######################################################################
stock_industry_map = {
    "TD.TO": "Banks",
    "RY.TO": "Banks",
    "BNS.TO": "Banks",
    "ABX.TO": "Mining",
    "BMO.TO": "Banks",
    "CM.TO": "Banks",
    "L.TO": "Food",
    "FNV.TO": "Unknown",
    "NA.TO": "Banks",
    "TRP.TO": "Pipelines",
    "CVE.TO": "Oil & Gas",
    "CNQ.TO": "Oil & Gas",
    "WN.TO": "Food",
    "SU.TO": "Oil & Gas",
    "ENB.TO": "Pipelines",
    "MG.TO": "Auto Parts & Equipment",
    "K.TO": "Mining",
    "IMO.TO": "Oil & Gas",
    "SLF.TO": "Insurance",
    "EMA.TO": "Electric",
    "DOL.TO": "Retail",
    "CP.TO": "Transportation",
    "MFC.TO": "Insurance",
    "MRU.TO": "Food",
    "POW.TO": "Insurance",
    "FVI.TO": "Unknown",
    "FTS.TO": "Electric",
    "CMG.TO": "Unknown",
    "DPM.TO": "Unknown",
    "GSY.TO": "Unknown",
    "FM.TO": "Mining",
    "WPM.TO": "Unknown",
    "BHC.TO": "Unknown",
}


######################################################################
# Data processing
######################################################################
def get_processed_data(df, period: str):
    try:
        if period == "Year":
            df["Year"] = pd.PeriodIndex(df.Date, freq="Y")
            df.drop("Date", axis=1, inplace=True)
            df.set_index("Year", inplace=True)
            df = df.pct_change().dropna()
        elif period == "Quarter":
            df["Quarter"] = pd.PeriodIndex(df.Date, freq="Q")
            df.drop("Date", axis=1, inplace=True)
            df.set_index("Quarter", inplace=True)
            df = df.pct_change().dropna()
        return df
    except Exception as e:
        print(f"Error found executing get_processed_data(): {e}")


######################################################################
# Function to get industry specific data / returns
######################################################################
def get_industry_details(df, industry: str, criteria: str, period: str):
    try:
        ind_list = []
        for st, ind in stock_industry_map.items():
            if ind == industry:
                ind_list.append(st)
            # else:
            #     ind_list = None #list(set(stock_industry_map.values()))
        if criteria == "mean":
            df_mean = df[ind_list].groupby(period).mean()
            df_mean.index = df_mean.index.astype(str)
            plot_stock_returns(df_mean, industry, period)
            return df_mean
        elif criteria == "max":
            df_max = df[ind_list].groupby(period).max()
            df_max.index = df_max.index.astype(str)
            plot_stock_returns(df_max, industry, period)
            return df_max
        elif criteria == "min":
            df_min = df[ind_list].groupby(period).min()
            df_min.index = df_min.index.astype(str)
            plot_stock_returns(df_min, industry, period)
            return df_min
    except Exception as e:
        print(f"Error found executing get_industry_details(): {e}")


######################################################################
# Plot the returns based on the industry
######################################################################
def plot_stock_returns(df, industry: str, period: str):
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in df.columns.values:
            ax.plot(df[i], label=i)
        ax.set_title(f"Industry Returns for {industry} - {period} analysis")
        ax.set_xlabel("Year", fontsize=18)
        ax.set_ylabel("Avg Annual Returns", fontsize=18)
        ax.legend(df.columns.values, loc="upper left")
        plt.grid()
        plt.show(fig)
    except Exception as e:
        print(f"Error found executing plot_stock_returns(): {e}")


######################################################################
# Stages
######################################################################
start_date = "2020-01-01"
end_date = "2023-01-01"


def get_Results(period: str, industry: str, measure: str):
    df_stock_price = get_data(start_date, end_date)
    get_basic_stats(df_stock_price)
    df_clean = get_clean_data(df_stock_price)
    df_processed = get_processed_data(df_clean, period)
    df_industry_details = get_industry_details(df_processed, industry, measure, period)


#############################################################################################
# Analysis and visualization of all combinations of 10 industry, 2 period and 3 returns types
#############################################################################################

import itertools
from itertools import product

# cart_list = list(
#     product(
#         ["Year", "Quarter"],
#         list(set(stock_industry_map.values())),
#         ["mean", "max", "min"],
#     )
# )
# [get_Results(item[0], item[1], item[2]) for i, item in enumerate(cart_list)]


######################################################################
# Annual returns and Value at Risk
######################################################################


def get_risk():
    try:
        lookback_period = 252
        var_level = 0.05  # 5% significance level
        df_stock_price = get_data()
        df_clean = get_clean_data(df_stock_price)
        df_clean.set_index("Date", inplace=True)
        df_Returns = df_clean.pct_change().dropna()
        var_data = df_Returns.tail(lookback_period)
        stock_volatility = df_Returns.std()

        VaR = []
        for stk in var_data:
            risk = abs(np.percentile(var_data[stk], var_level * 100) * 100)
            VaR.append(round(risk, 3))

        df_risk = pd.DataFrame(
            {
                "Stock Returns": df_Returns.columns,
                "Value at Risk (vaR)": VaR,
                "Volatility": stock_volatility,
            }
        )
        df_risk = df_risk.sort_values("Value at Risk (vaR)", ascending=False)
        df_risk.set_index("Stock Returns", inplace=True)

        plt.figure(figsize=(12, 25))
        df_risk.sort_values("Value at Risk (vaR)", inplace=True)
        df_risk.plot(kind="barh", y="Value at Risk (vaR)", color="b")
        plt.show()

        plt.figure(figsize=(12, 25))
        df_risk.sort_values("Volatility", inplace=True)
        df_risk.plot(kind="barh", y="Volatility", color="b")
        plt.show()

        return df_risk
    except Exception as e:
        print(f"Error found executing get_risk(): {e}")


######################################################################
# Industry wise risk and performance
######################################################################


def get_industry_risk():
    try:
        df_industry_risk["Industry"] = [
            stock_industry_map[col]
            for col in df_industry_risk.index
            if col in stock_industry_map.keys()
        ]

        df_volatility = get_risk()
        df_industry_risk["Volatility"] = df_volatility["Volatility"]

        df_industry_risk_agg = df_industry_risk.groupby("Industry")[
            "Value at Risk (vaR)"
        ].mean()

        print(
            df_industry_risk.groupby(["Industry", "Stock Returns"])[
                "Value at Risk (vaR)"
            ].mean()
        )
        print(
            df_industry_risk.groupby(["Industry", "Stock Returns"])["Volatility"].mean()
        )

        df_industry_volatility_agg = df_industry_risk.groupby("Industry")[
            "Volatility"
        ].mean()

        plt.figure(figsize=(8, 6))
        df_industry_risk_agg.sort_values(inplace=True)
        df_industry_risk_agg.plot(kind="barh", y="Industry", color="b")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 6))
        df_industry_volatility_agg.sort_values(inplace=True)
        df_industry_volatility_agg.plot(kind="barh", y="Industry", color="b")
        plt.legend()
        plt.grid()
        plt.show()

        return df_industry_risk_agg, df_industry_risk
    except Exception as e:
        print(f"Error found executing get_industry_risk(): {e}")


######################################################################
# Execution
######################################################################
get_Results("Year", "Banks", "mean")
get_Results("Quarter", "Banks", "mean")

get_Results("Year", "Food", "mean")
get_Results("Quarter", "Food", "mean")


df_industry_risk = get_risk()

df_vol = get_industry_risk()
