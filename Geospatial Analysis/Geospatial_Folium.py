import os
import matplotlib.pyplot as plt
import geopandas
import folium
import pandas as pd
import numpy as np


###############################################################
# TORNADOES ANALYSIS
###############################################################
states = geopandas.read_file("usa-states-census-2014.shp")
# http://www.spc.noaa.gov/gis/svrgis/zipped/1950-2018-torn-initpoint.zip
tornado = geopandas.read_file("1950-2018-torn-initpoint.shp")

tornado_states_filtered = tornado[tornado["st"].isin(list(states.STUSPS.unique()))]


#########################################################
# FOLIUM
#########################################################

from branca.element import Figure

map_initial = folium.Map(location=[40, -95], zoom_start=4)
map_initial

# fig=Figure(width=550,height=350)
map1 = folium.Map(
    location=[33.812092, -117.918976], zoom_start=14
)  # ,min_zoom=8,max_zoom=14)
map1

# Adding markers to the map
marker_popup = folium.Marker(
    location=[33.812092, -117.918976],
    popup="<stong>Address: Disney Land, Anaheim, CA 92802, USA</stong>",
    tooltip="Click here for more info",
)
marker_popup.add_to(map1)
map1


# Adding circle markers to the map
marker_circle = folium.CircleMarker(
    location=[33.812092, -117.918976],
    popup="<stong>Address: Disney Land, Anaheim, CA 92802, USA</stong>",
    tooltip="Click here for more info",
    color="blue",
    fill=True,
    radius=50,
)
marker_circle.add_to(map1)
map1

folium.TileLayer("openstreetmap").add_to(map1)
folium.TileLayer("Stamen Terrain").add_to(map1)
folium.TileLayer("Stamen Toner").add_to(map1)
folium.TileLayer("Stamen Water Color").add_to(map1)
folium.TileLayer("cartodbpositron").add_to(map1)
folium.TileLayer("cartodbdark_matter").add_to(map1)
folium.LayerControl().add_to(map1)

map1


#######################################################################################
# Tornado mapping
#######################################################################################
tornado_states_filtered["cnt"] = 1
tornado_states_filtered.columns
tornado_states_filtered["lon"] = tornado_states_filtered["geometry"].x
tornado_states_filtered["lat"] = tornado_states_filtered["geometry"].y
tornado_states_filtered["LatLong"] = list(
    zip(tornado_states_filtered.lat, tornado_states_filtered.lon)
)
tornado_states_filtered["LatLong"] = [
    list(tp) for tp in tornado_states_filtered["LatLong"]
]
tornado_states_filtered["LatLong_cnt"] = list(
    zip(
        tornado_states_filtered.lat,
        tornado_states_filtered.lon,
        tornado_states_filtered.cnt,
    )
)


# fig=Figure(width=550,height=350)

map2 = folium.Map(
    location=[33.812092, -117.918976], zoom_start=5
)  # ,min_zoom=8,max_zoom=14)

latlon = tornado_states_filtered["LatLong"].head(1000)
for coord in latlon:
    # print(coord)
    map2 = folium.Map(location=coord)  # , zoom_start = 5)#,min_zoom=8,max_zoom=14)

folium.TileLayer("Stamen Terrain").add_to(map2)
folium.TileLayer("Stamen Toner").add_to(map2)
folium.TileLayer("Stamen Water Color").add_to(map2)
folium.TileLayer("cartodbpositron").add_to(map2)
folium.TileLayer("cartodbdark_matter").add_to(map2)
folium.LayerControl().add_to(map2)


map2


#######################################################################################
# Tornado and heatmap
#######################################################################################

import folium.plugins as plugins
from folium.plugins import HeatMap


folium_heatmap = folium.Map(location=[33.812092, -117.918976], zoom_start=5)
folium_heatmap.add_child(
    plugins.HeatMap(tornado_states_filtered["LatLong_cnt"], radius=15)
)

folium_heatmap


#######################################################################################
# Tornado and Choropleth map
#######################################################################################

state_level_tornadoes = (
    tornado_states_filtered[["st", "cnt"]].groupby("st").count().reset_index()
)

map4 = folium.Map(location=[48, -102], zoom_start=3)

url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data"
state_geo = f"{url}/us-states.json"


folium.Choropleth(
    geo_data=state_geo,
    name="choropleth",
    data=state_level_tornadoes,
    columns=["st", "cnt"],
    key_on="feature.id",
    fill_color="Paired",
    fill_opacity=0.75,
    line_opacity=0.3,
    legend_name="Impact of Tornadoes",
).add_to(map4)

map4

folium.LayerControl().add_to(map4)

map4


#######################################################################################
# Folium - Minimap
#######################################################################################


from folium.plugins import MiniMap

minimap = MiniMap()
map4.add_child(minimap)
map4
