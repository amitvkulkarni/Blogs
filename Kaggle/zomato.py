import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# pd.set_option('display.max_rows', None)
# pd.reset_option("^display.", silent=True)
######################################################################
# Load Data
######################################################################
df_zomato = pd.read_csv("./zomato_dataset.csv", skipinitialspace=True)
df_zomato.columns = df_zomato.columns.str.replace(" ", "")


df_zomato["DeliveryRating"] = df_zomato["DeliveryRating"].fillna(0)
df_zomato["DiningRating"] = df_zomato["DeliveryRating"].fillna(0)

######################################################################
# Basic EDA
######################################################################
df_zomato.shape

df_zomato.columns

df_zomato.head()

df_zomato.info()

df_zomato.describe()

df_zomato.isnull().sum()

######################################################################
# City wise analysis
######################################################################

# Number of places in the city
df_zomato.groupby("City")["PlaceName"].nunique()

# Most expensive Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].max()
idx = df_city.groupby("City")["Prices"].idxmax()
df_city_max_price = df_city.loc[idx]


# Least expensive Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].min()
idx = df_city.groupby("City")["Prices"].idxmin()
df_city_min_price = df_city.loc[idx]

# Average price of  Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].mean()
idx = df_city.groupby("City")["Prices"].idxmin()
df_city_avg_price = df_city.loc[idx]

# Most expensive ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].max()
idx = df_city.groupby("City")["Prices"].idxmax()
df_city_max_item = df_city.loc[idx]


# Least expensive ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].min()
idx = df_city.groupby("City")["Prices"].idxmin()
df_city_min_item = df_city.loc[idx]

# Average price of ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].mean()
idx = df_city.groupby("City")["Prices"].idxmin()
df_city_avg_price = df_city.loc[idx]


# Most popular Restaurant by city
df_zomato["Total_rating"] = df_zomato["DiningRating"] + df_zomato["DeliveryRating"]
df_rating = df_zomato.groupby(["City", "RestaurantName"], as_index=False)[
    "Total_rating"
].max()
idx = df_rating.groupby("City")["Total_rating"].idxmax()
df_rating_max = df_rating.loc[idx]
df_rating_max = df_rating_max.set_index("City")


df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].max()


def get_plots(df, criteria, title):
    list_cities = list(df.City.unique())
    my_colors = ["#FA8072", "#6495ED", "#40E0D0", "#808080", "#28B463"]
    fig, axs = plt.subplots(17, 1, figsize=(10, 60), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    axs = axs.ravel()
    for i, city in enumerate(list_cities):
        df_filtered = df[df["City"] == city]
        df_filtered.sort_values("Prices", ascending=False, inplace=True)
        df_filtered = df_filtered.head(5)
        bars = axs[i].barh(
            df_filtered[criteria], df_filtered["Prices"], color=my_colors
        )
        axs[i].set_title(f"{title} {criteria} in {city}")
        axs[i].bar_label(bars)


# Most expensive Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].max()
get_plots(df_city, "Cuisine", "Top 5 most expensive ")

# Average price of top 5 Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].mean()
get_plots(df_city, "Cuisine", "Average price of top 5 Cuisine ")

# Prices of lowest priced Cuisine by city
df_city = df_zomato.groupby(["City", "Cuisine"], as_index=False)["Prices"].min()
get_plots(df_city, "Cuisine", "5 lowest priced Cuisine by city ")


# Most expensive ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].max()
get_plots(df_city, "ItemName", "Top 5 most expensive ")


# Prices of lowest priced ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].min()
get_plots(df_city, "ItemName", "5 lowest priced Cuisine by city ")

# Average price of top 5 ItemName by city
df_city = df_zomato.groupby(["City", "ItemName"], as_index=False)["Prices"].mean()
get_plots(df_city, "ItemName", "Average price of top 5 ItemName ")


# 1. Distribution of ratings
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].hist(df_zomato["DiningRating"], edgecolor="black")
ax[1].hist(df_zomato["DeliveryRating"], edgecolor="black")
# plot 2 subplots
ax[0].set_title("Distribution of Dining Ratings")
ax[1].set_title("Distribution of Delivery Ratings")
fig.suptitle("Distribution of Ratings")
plt.tight_layout()
plt.show()


# 2. Distribution of votes
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].hist(df_zomato["DiningVotes"], edgecolor="black")
ax[1].hist(df_zomato["Votes"], edgecolor="black")
# plot 2 subplots
ax[0].set_title("Distribution of Dining votes")
ax[1].set_title("Distribution of votes")
fig.suptitle("Distribution of Votes")
plt.tight_layout()
plt.show()


# Total number of outlets for each brand.

res_cnt = df_zomato["RestaurantName"].value_counts().head(20)
plt.figure(figsize=(12, 6))
ax = res_cnt.plot(kind="bar")
ax.legend(["* Restaurants"])
plt.xlabel("Name of Restaurant")
plt.ylabel("Count of Restaurants")
plt.title("Name vs Number of Restaurant", fontsize=20, weight="bold")

# Cities with maximum restaurants
num_rest = df_zomato["City"].value_counts().nlargest(12).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
ax = num_rest.plot(kind="barh", color="#6495ED")
plt.xlabel("Count of Restaurant")
plt.ylabel("Name of Restaurants")
plt.title("Number of Restaurant in the cities", fontsize=8, weight="bold")
ax.invert_yaxis()
plt.tight_layout()


# Does higher price mean higher ratings?
plt.scatter(df_zomato["DiningRating"], np.log(df_zomato["Prices"]), alpha=0.25)
plt.xlabel("Dining Rating")
plt.ylabel("Prices")
plt.title("Price Vs Dining Rating", fontsize=8, weight="bold")
plt.show()


# Analysis of best sellers and other categories
df_bestseller = (
    df_zomato["BestSeller"].value_counts().nlargest(5).sort_values(ascending=False)
)
my_colors = ["#FA8072", "#6495ED", "#40E0D0", "#808080", "#28B463"]

plt.figure(figsize=(12, 6))
plt.pie(
    df_bestseller,
    colors=my_colors,
    labels=df_bestseller.index,
    autopct="%1.1f%%",
    pctdistance=0.85,
)
centre_circle = plt.Circle((0, 0), 0.5, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("% share of Top 5 Best Sellers category")
# plt.legend()
plt.show()


# Top 5 restaurants in each city
def get_restaurant_plots(df, title):
    list_cities = list(df.City.unique())
    print(f"***********************************************{len(list_cities)}")
    my_colors = ["#FA8072", "#6495ED", "#40E0D0", "#808080", "#28B463"]
    fig, axs = plt.subplots(
        len(list_cities), 1, figsize=(10, 60), facecolor="w", edgecolor="k"
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    axs = axs.ravel()
    for i, city in enumerate(list_cities):
        df_filtered = df[df["City"] == city]
        df_filtered.sort_values("count", ascending=False, inplace=True)
        df_filtered = df_filtered.head(5)
        bars = axs[i].barh(
            df_filtered["RestaurantName"], df_filtered["count"], color=my_colors
        )
        axs[i].set_title(f"{title} in {city}")
        axs[i].bar_label(bars)


df_res_cnt = df_zomato.groupby(["City", "RestaurantName"], as_index=False)[
    "RestaurantName"
].value_counts()
get_restaurant_plots(df_res_cnt, "Top 5 restaurants in  ")


# Merge various areas of Bangalore under Bangalore
BLR_areas = ["Banaswadi", "Ulsoor", "Malleshwaram", "Magrath Road"]
df_BLR = df_zomato.copy()
df_BLR["City"].replace(BLR_areas, "Bangalore", inplace=True)
df_BLR_cnt = df_BLR.groupby(["City", "RestaurantName"], as_index=False)[
    "RestaurantName"
].value_counts()
get_restaurant_plots(df_BLR_cnt, "Top 5 restaurants in  ")
