import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import numpy as np



# states = geopandas.read_file("usa-states-census-2014.shp")
states = geopandas.read_file("C:/Users/kulkarna4029/OneDrive - ARCADIS/Desktop/Blogs/Geo Spatial/geopandas-tutorial-master/geopandas-tutorial-master/data/usa-states-census-2014.shp")


print(states.shape)
print(states.columns)
states.head()
states[['STUSPS','NAME', 'region', 'geometry']].head()

# check for CRS
states.crs

# states = states.to_crs("EPSG:3395")
# torn = torn.to_crs("EPSG:3395")

states.plot()
states.boundary.plot()
states.plot(cmap='magma', figsize=(12, 12))



fig, axes = plt.subplots(2,2, figsize=(12,6), sharex=True, sharey=True)
base = states.plot(ax=axes[0,0])
axes[0,0].set_title("Standard plot")

base = states.boundary.plot(ax=axes[0,1])
axes[0,1].set_title("Boundary plot")

base = states.plot(cmap = "magma",ax=axes[1,0])
axes[1,0].set_title("magma Theme")

base = states.plot(cmap = "Pastel1",ax=axes[1,1])
axes[1,1].set_title("Paste1 theme")




#####################################################################
# Specific state analysis and plotting - california
#####################################################################
states[states['NAME'] == 'California'].plot(figsize = (5,5))


midwest = states[states['region'] == 'Midwest']

fig = plt.figure(1, figsize=(12,12)) 
ax = fig.add_subplot()
states.apply(lambda x: ax.annotate(text = str(x.STUSPS), xy=x.geometry.centroid.coords[0], ha='center', fontsize=14),axis=1)
states.boundary.plot(ax=ax, color='Black', linewidth=.4)
states.plot(ax=ax, color='ivory', figsize=(12, 12))
# midwest.plot(ax=ax, color='orangered',alpha = 0.5, marker='.', markersize=1)
midwest.plot(ax=ax)
ax.text(-0.05, 0.5, transform=ax.transAxes,
        fontsize=20, color='gray', alpha=0.2,
        ha='center', va='center', rotation='90')

#####################################################################
# Region wise analysis and plotting with color codes
#####################################################################


west = states[states['region'] == 'West']
southwest = states[states['region'] == 'Southwest']
southeast = states[states['region'] == 'Southeast']
midwest = states[states['region'] == 'Midwest']
northeast = states[states['region'] == 'Northeast']

fig = plt.figure(1, figsize=(12,12)) 
ax = fig.add_subplot()
states.apply(lambda x: ax.annotate(text = str(x.STUSPS), xy=x.geometry.centroid.coords[0], ha='center', fontsize=14),axis=1)
states.boundary.plot(ax=ax, color='Black', linewidth=.4)
# states.plot(ax=ax, color='ivory', figsize=(12, 12))
southwest.plot(ax=ax)
west.plot(ax=ax,color="MistyRose")
southwest.plot(ax=ax, color="PaleGoldenRod")
southeast.plot(ax=ax, color="Plum")
midwest.plot(ax=ax, color="PaleTurquoise")
northeast.plot(ax=ax, color="ivory")
ax.text(-0.05, 0.5, transform=ax.transAxes,
        fontsize=20, color='gray', alpha=0.2,
        ha='center', va='center', rotation='90')



###############################################################
# TORNADOES ANALYSIS with Matplotlib and GeoPandas
###############################################################

# http://www.spc.noaa.gov/gis/svrgis/zipped/1950-2018-torn-initpoint.zip
torn = geopandas.read_file("1950-2018-torn-initpoint.shp")



# Basic EDA
torn.head()
torn.shape
torn.columns

# check for CRS
torn.crs


len(list(torn.st.unique()))
len(list(states.STUSPS.unique()))

torn.plot( figsize=(12,9), color='red', marker='x', markersize=1)

torn_states_filtered = torn[torn['st'].isin(list(states.STUSPS.unique()))]

torn_states_filtered.plot( figsize=(12,9), color='red', marker='x', markersize=1)


# visualization on the map
fig = plt.figure(1, figsize=(25,15)) 
ax = fig.add_subplot()
states.apply(lambda x: ax.annotate(text = str(x.STUSPS), xy=x.geometry.centroid.coords[0], ha='center', fontsize=14),axis=1)
states.boundary.plot(ax=ax, color='black', linewidth=.4)
states.plot(ax=ax, color = 'ivory', figsize=(12, 12))
torn_states_filtered.plot(ax=ax, color='red',alpha = 0.5, marker='.', markersize=1)
ax.text(-0.05, 0.5, transform=ax.transAxes,
        fontsize=20, color='white', alpha=0.2,
        ha='center', va='center', rotation='90')



torn['cnt'] = 1
torn_sub = torn[['st','cnt']].groupby('st').count()

torn_sub = torn_sub.sort_values('cnt', ascending=True).tail(20)
torn_sub.plot.barh()




