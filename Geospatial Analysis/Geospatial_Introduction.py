import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


#####################################################################
# Load the world countries shapefile from GeoPandas datasets
#####################################################################

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(10, 8))
# world.plot(ax=ax, cmap = 'OrRd', color='lightgrey')
world.plot(ax=ax, cmap = 'OrRd')
plt.show()


#states = gpd.read_file("IND_adm/IND_adm3.shp")
Belgium = gpd.read_file("BEL_adm/BEL_adm1.shp")
Belgium.head()


# plots the map
Belgium.plot(cmap='viridis', figsize=(5, 5))


#####################################################################
# Fetch the territory names and annotate on the map
#####################################################################

Belgium["coords"] = Belgium["geometry"].apply(lambda x: x.representative_point().coords[:])
Belgium["coords"] = [coords[0] for coords in Belgium["coords"]]

fig, ax = plt.subplots(figsize = (6,6))
# Belgium.plot(ax=ax, color="yellow", edgecolor="black")
Belgium.plot(ax=ax, cmap='viridis')

for idx, row in Belgium.iterrows():
    print(f'**********************{row["NAME_0"]}')
    plt.annotate(text=row["NAME_1"], xy=row["coords"], 
                horizontalalignment="center", color = "purple")

plt.show()


Belgium = gpd.read_file("BEL_adm/BEL_adm3.shp")
state = Belgium[Belgium['NAME_1'] == 'Wallonie']
# state.plot(figsize=(8, 8), color = "Yellow")
state.plot(cmap='viridis', figsize=(5, 5))


#####################################################################
# Download earthquake data from USGS
#####################################################################
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
earthquake_data = pd.read_csv(url)

# Create a GeoDataFrame from the earthquake data by converting latitude and longitude to Point geometries
geometry = gpd.points_from_xy(earthquake_data.longitude, earthquake_data.latitude)
earthquakes = gpd.GeoDataFrame(earthquake_data, geometry=geometry)

# Plot the world map along with earthquake locations
fig, ax = plt.subplots(figsize=(10, 8))
world.plot(ax=ax,  color='lightgrey')
# plt.show()
earthquakes.plot(ax=ax, markersize=earthquakes['mag']*2, color='red', alpha=0.5)
plt.title("Earthquake Activity Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()