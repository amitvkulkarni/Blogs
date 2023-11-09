import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
import folium

# pd.set_option("display.max_rows", None)


#########################################################
# Data Loading
#########################################################
df_house_price = pd.read_csv("./data/AB_NYC_2019.csv")
df_house_price.head()

#########################################################
# Data exploration
#########################################################

df_house_price.info()

df_house_price.describe()

# data types
df_house_price.dtypes

# Missing values
df_house_price.isnull().sum()

# Percentage of missing values for each column
round(df_house_price.isnull().sum() / len(df_house_price) * 100, 2)

# Max price for neighbourhood_group
df_house_price.groupby("neighbourhood_group")["price"].max()

df_house_price.groupby("neighbourhood_group")["price"].count()

# # Mean price for neighbourhood_group
df_house_price.groupby("neighbourhood_group")["price"].mean()


df_house_price.groupby(["neighbourhood_group", "neighbourhood"]).agg(
    properties=("host_id", "count")
)

#########################################################
# Analysis of price by group and room type
#########################################################
df_type_price = df_house_price.groupby(["neighbourhood_group", "room_type"]).agg(
    price=("price", "mean")
)
df_type_price = df_type_price.reset_index()
sns.barplot(x="neighbourhood_group", y="price", hue="room_type", data=df_type_price)

#####################################################################
# Analysis of number of room of each type across neighbourhood_group
#####################################################################
df_room_type = df_house_price.groupby(["neighbourhood_group", "room_type"]).agg(
    count=("room_type", "count")
)
df_room_type = df_room_type.reset_index()
sns.barplot(x="neighbourhood_group", y="count", hue="room_type", data=df_room_type)


#########################################################
# Price trends and neighbourhood groups
#########################################################
# df_room_nights = df_house_price.groupby(
#     ["neighbourhood_group", "room_type", "minimum_nights"]
# )["price"].mean()
# df_room_nights = df_room_nights.reset_index()


df_room_nights = df_house_price.groupby(["neighbourhood_group", "room_type"]).agg(
    nights=("minimum_nights", "mean")
)
df_room_nights = df_room_nights.reset_index()

sns.barplot(x="neighbourhood_group", y="nights", hue="room_type", data=df_room_nights)


#########################################################
# Calculating the average price per region
#########################################################
df_group_avg_price = round(
    df_house_price.groupby("neighbourhood_group")
    .price.mean()
    .sort_values(ascending=False),
    2,
)

# Plotting the average price per region
fig = plt.figure(1, figsize=(6, 4))
plt.bar(df_group_avg_price.index, df_group_avg_price)
plt.title("Property listing for each region")
plt.xlabel("Region")
plt.ylabel("Average pricing per region")
plt.show()


#########################################################
# Calculating the maximum price per region
#########################################################
df_group_max_price = round(
    df_house_price.groupby("neighbourhood_group")
    .price.max()
    .sort_values(ascending=False),
    2,
)

fig = plt.figure(1, figsize=(6, 4))
# Plotting the average price per region
plt.bar(df_group_max_price.index, df_group_max_price)
plt.title("Property listing for each region")
plt.xlabel("Region")
plt.ylabel("Max pricing per region")
plt.show()


#########################################################
# Calculating the number of properties per region
#########################################################
df_group_region_count = round(
    df_house_price.groupby("neighbourhood_group")
    .price.count()
    .sort_values(ascending=False),
    2,
)

fig = plt.figure(1, figsize=(6, 4))
# Plotting the average price per region
plt.bar(df_group_region_count.index, df_group_region_count)
plt.title("Property listing for each region")
plt.xlabel("Region")
plt.ylabel("Number of properties per region")
plt.show()


##########################################################################
# Geo spatial analysis
##########################################################################

# basic scatterplot
df_house_price.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    c=df_house_price["price"],
    s=8,
    cmap=plt.get_cmap("jet"),
    figsize=(8, 6),
)


nyc_roads = geopandas.read_file("./data/nyu_2451_34499 - City Roads/nyu_2451_34499.shp")
nyc_roads.plot()
nyc_roads.explore(legend=False)

nyc_boroughs = geopandas.read_file(
    "./data/nyu_2451_34154- Borough Boundaries/nyu_2451_34154.shp"
)

nyc_boroughs.plot("SHAPE_AREA", legend=False)

# pip install mapclassify
nyc_boroughs.explore("SHAPE_AREA", legend=False)


##########################################################################
# Price analysis across regions
##########################################################################


def plot_NYC_Price_Map(df):
    # Create a map centered around NYC
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    data = df[["neighbourhood", "latitude", "longitude", "name", "host_name", "price"]]

    # Add markers for data points
    for index, row in data.iterrows():
        tooltip = f"Property Name: {row['name']}, Price:  {row['price']}, Host Name: {row['host_name']}, Location: {row['latitude'], {row['longitude']}}"
        # tooltip = f"{row['name']}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=row["neighbourhood"],
            tooltip=tooltip,
        ).add_to(nyc_map)

    # Color the boroughs using Choropleth
    folium.Choropleth(
        # geo_data=nyc_boroughs,
        geo_data=nyc_roads,
        fill_color="green",
        fill_opacity=0.2,
        line_opacity=0.2,
    ).add_to(nyc_map)

    return nyc_map


################################################################
# Plotting all the properties across region between price range
################################################################

LOW_PRICE = 5000
HIGH_PRICE = 10000

df_high_price = df_house_price[df_house_price["price"].between(LOW_PRICE, HIGH_PRICE)]
all_regions = plot_NYC_Price_Map(df_high_price)
all_regions


df_high_price_Queens = df_house_price[
    (df_house_price["neighbourhood_group"] == "Queens")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
Queens = plot_NYC_Price_Map(df_high_price_Queens)
Queens

df_high_price_Brooklyn = df_house_price[
    (df_house_price["neighbourhood_group"] == "Brooklyn")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
Brooklyn = plot_NYC_Price_Map(df_high_price_Brooklyn)
Brooklyn

df_high_price_Manhattan = df_house_price[
    (df_house_price["neighbourhood_group"] == "Manhattan")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
Manhattan = plot_NYC_Price_Map(df_high_price_Manhattan)
Manhattan

df_high_price_StatenIsland = df_house_price[
    (df_house_price["neighbourhood_group"] == "Staten Island")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
StatenIsland = plot_NYC_Price_Map(df_high_price_StatenIsland)
StatenIsland

df_high_price_Bronx = df_house_price[
    (df_house_price["neighbourhood_group"] == "Bronx")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
Bronx = plot_NYC_Price_Map(df_high_price_Bronx)
Bronx

################################################################
# Plotting 50 properties with lowest price and the highest price
################################################################

lowest50 = plot_NYC_Price_Map(
    df_house_price.sort_values(by="price", ascending=True).head(50)
)
lowest50

highest50 = plot_NYC_Price_Map(
    df_house_price.sort_values(by="price", ascending=False).head(50)
)
highest50.save("high.html")


################################################################
# Plotting 50 properties with the highest price for all regions
################################################################

df_high_price_Queens = df_house_price[df_house_price["neighbourhood_group"] == "Queens"]
df_high_price_Queens = df_high_price_Queens.sort_values(
    by="price", ascending=False
).head(50)
Queens = plot_NYC_Price_Map(df_high_price_Queens)
Queens
Queens.save("Queens.html")


df_high_price_Brooklyn = df_house_price[
    df_house_price["neighbourhood_group"] == "Brooklyn"
]
df_high_price_Brooklyn = df_high_price_Brooklyn.sort_values(
    by="price", ascending=False
).head(50)
Brooklyn = plot_NYC_Price_Map(df_high_price_Brooklyn)
Brooklyn
Brooklyn.save("Brooklyn.html")


df_high_price_StatenIsland = df_house_price[
    df_house_price["neighbourhood_group"] == "Staten Island"
]
df_high_price_StatenIsland = df_high_price_StatenIsland.sort_values(
    by="price", ascending=False
).head(50)
StatenIsland = plot_NYC_Price_Map(df_high_price_StatenIsland)
StatenIsland
StatenIsland.save("StatenIsland.html")

df_high_price_Bronx = df_house_price[
    (df_house_price["neighbourhood_group"] == "Bronx")
    & (df_house_price["price"].between(LOW_PRICE, HIGH_PRICE))
]
Bronx = plot_NYC_Price_Map(df_high_price_Bronx)
Bronx


##########################################################################
# Identifying the room types with different icons
##########################################################################


df_high_price_manhattan = df_house_price[
    df_house_price["neighbourhood_group"] == "Manhattan"
]
df_high_price_manhattan = df_high_price_manhattan.head(100)
# df_high_price_manhattan = df_high_price_manhattan.sort_values(
#     by="price", ascending=False
# ).head(50)

map_manhattan = folium.Map(location=[40.7831, -73.9712], zoom_start=13)

# df_manhattan = df_house_price[df_house_price['neighbourhood_group'] == "Manhattan"].head(50)
df_high_price_manhattan["room_type"].value_counts()

# Add markers for each room type
for index, row in df_high_price_manhattan.iterrows():
    tooltip = f"Property Name: {row['name']}, Price:  {row['price']}, Room Type:  {row['room_type']}, Host Name: {row['host_name']}, Location: {row['latitude'], {row['longitude']}}"

    if row["room_type"] == "Private room":
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=folium.Icon(color="blue"),
            popup=row["room_type"],
            tooltip=tooltip,
        ).add_to(map_manhattan)
    elif row["room_type"] == "Shared room":
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=folium.Icon(color="green"),
            popup=row["room_type"],
            tooltip=tooltip,
        ).add_to(map_manhattan)
    elif row["room_type"] == "Entire home/apt":
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=folium.Icon(color="red"),
            popup=row["room_type"],
            tooltip=tooltip,
        ).add_to(map_manhattan)

folium.Choropleth(
    # geo_data=nyc_boroughs,
    geo_data=nyc_roads,
    # fill_color="green",
    fill_opacity=0.2,
    line_opacity=0.2,
).add_to(map_manhattan)

map_manhattan

map_manhattan.save("manhattan.html")
