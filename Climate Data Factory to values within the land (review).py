#!/usr/bin/env python
# coding: utf-8

# In[48]:



import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from itertools import product
from cftime import DatetimeNoLeap


# In[49]:


path = 'C:/Users/kkwan/Downloads/prAdjust-Indices_average_GFDL-ESM2M_rcp85_r1i1p1_gr025_TCDF-CDFT23-ERA5-1981-2010_2051-2080.nc'
df = xr.open_mfdataset(path,combine='by_coords')


# In[50]:


df


# In[51]:


ds = df.to_dataframe()
ds = ds.reset_index()


# In[52]:


ds.head(3)


# In[53]:


#shift = lambda arg : arg  - 45
#ds['lon_bnds'] = ds['lon_bnds'].apply(shift)

#shift map to the correct area
shift = lambda T: T - 360  if T>180  else T


# In[54]:


ds['lon_bnds']=ds['lon_bnds'].apply(shift)


# In[55]:


#formats date
ds['time_bnds_string'] = ds['time_bnds'].astype('str')
ds["time_bnds_date"] = pd.to_datetime(ds["time_bnds_string"],format= '%Y-%m-%d %H:%M:%S',errors='coerce')
     


# In[66]:


#selects the time frame and index
date_old = '2080-12-31 12:00:00'
date_new  = '2051-12-31 12:00:00'

col = 'consecutive_dry_days_index_per_time_period'
col1 = 'very_heavy_precipitation_days_index_per_time_period'


# In[67]:


ds.tail(3)


# In[68]:



#lower_date = datetime.datetime.strptime(lower_date,'%m/%d/%Y')
date_1 = dt.datetime.strptime(date_old,'%Y-%m-%d %H:%M:%S')
date_2 = dt.datetime.strptime(date_new,'%Y-%m-%d %H:%M:%S')

ds_old = ds[(ds['time_bnds_date'] == date_old)]
ds_old= ds_old.reset_index(drop=True)

ds_new = ds[(ds['time_bnds_date'] == date_new)]
ds_new= ds_new.reset_index(drop=True)
     


# In[69]:


ds_new.head(3)


# In[70]:


data=ds_old[['lat_bnds','lon_bnds',col]]
data = data.dropna()
data= data.reset_index(drop=True)

data1=ds_old[['lat_bnds','lon_bnds',col1]]
data1 = data1.dropna()
data1= data1.reset_index(drop=True)

data2=ds_new[['lat_bnds','lon_bnds',col1]]
data2 = data2.dropna()
data2= data2.reset_index(drop=True)


# In[71]:



data_value = data.values.tolist()
data_value1 = data1.values.tolist()
data_value2 = data2.values.tolist()


# In[72]:



#import gdal
from osgeo import gdal
import folium
from folium import plugins
from folium.plugins import HeatMap   
import pandas as pd
import numpy as n
from folium import FeatureGroup, LayerControl, Map, Marker


# In[73]:


#selects cities that we want to add
#data_city = {'City':['London', 'Paris', 'Germany', 'Rome','Italy','Miami','Niger','Oman','Australia','Hong Kong','Brighton','Quito']}

data_city = {'City' :[
    'Abu Dhabi',
'Andorra',
'Angola',
'Argentina',
'Armenia',
'Aruba',
'Australia',
'Austria',
'Azerbaijan',
'Bahrain',
'Bangladesh',
'Belarus',
'Belgium',
'Benin',
'Bolivia',
'Brazil',
'Bulgaria',
'Cameroon',
'Canada',
'Cabo Verde',
'Chile',
'China',
'Colombia',
'Congo, Republic of',
'Costa Rica',
#'Cote d'Ivoire',
'Croatia',
'Cyprus',
'Czech Republic',
'Denmark',
'Dominican Republic',
'Ecuador',
'Egypt',
'El Salvador',
'Estonia',
'Ethiopia',
'Finland',
'France',
'Gabon',
'Georgia',
'Germany',
'Ghana',
'Greece',
'Guatemala',
'Hong Kong',
'Hungary',
'Iceland',
'India',
'Indonesia',
'Iraq',
'Ireland',
'Israel',
'Italy',
'Jamaica',
'Japan',
'Hashemite Kingdom of Jordan',
'Kazakhstan',
'Kenya',
'Korea',
'Kuwait',
'Laos',
'Latvia',
'Lebanon',
'Lesotho',
'Lithuania',
'Luxembourg',
'Macao',
# 'Malaysia',
# 'Maldives',
# 'Malta',
# 'Mexico',
# 'Mongolia',
# 'Morocco',
# 'Mozambique',
# 'Namibia',
# 'Netherlands',
# 'New Zealand',
# 'Nicaragua',
# 'Nigeria',
# 'North Macedonia',
# 'Norway',
# 'Oman',
# 'Pakistan',
# 'Panama',
# 'Paraguay',
# 'Peru',
# 'Philippines',
# 'Poland',
# 'Portugal',
# 'Qatar',
# 'Ras Al Khaimah',
# 'Romania',
# 'Russia',
# 'Rwanda',
# 'San Marino',
# 'Saudi Arabia',
# 'Serbia',
# 'Seychelles',
# 'Singapore',
# 'Slovakia',
# 'Slovenia',
# 'South Africa',
# 'Spain',
# 'Sri Lanka',
# 'Suriname',
# 'Sweden',
# 'Switzerland',
# 'Taiwan',
# 'Thailand',
'Tunisia',
'Turkey',
'Uganda',
'Ukraine',
'United Kingdom',
'United States of America',
'Uruguay',
'Uzbekistan',
'Vietnam',
'Zambia']}


# In[74]:



# Convert the dictionary into DataFrame  
df = pd.DataFrame(data_city)
     


# In[75]:


#Gets the values into lat and lon

from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
   
# declare an empty list to store
# latitude and longitude of values 
# of city column
longitude = []
latitude = []
   
# function to find the coordinate
# of a given city 
def findGeocode(city):
       
    # try and catch is used to overcome
    # the exception thrown by geolocator
    # using geocodertimedout  
    try:
          
        # Specify the user_agent as your
        # app name it should not be none
        geolocator = Nominatim(user_agent="your_app_name")
          
        return geolocator.geocode(city)
      
    except GeocoderTimedOut:
          
        return findGeocode(city)    
  
# each value from city column
# will be fetched and sent to
# function find_geocode   
for i in (df["City"]):
      
    if findGeocode(i) != None:
           
        loc = findGeocode(i)
          
        # coordinates returned from 
        # function is stored into
        # two separate list
        latitude.append(loc.latitude)
        longitude.append(loc.longitude)
       
    # if coordinate for a city not
    # found, insert "NaN" indicating 
    # missing value 
    else:
        latitude.append(np.nan)
        longitude.append(np.nan)


# In[76]:




# now add this column to dataframe
df["Longitude"] = longitude
df["Latitude"] = latitude
     


# In[77]:



# shift = lambda arg : arg + 360
# df['Longitude'] = df['Longitude'].apply(shift)
locations = df[['Latitude', 'Longitude']]
locationlist = locations.values.tolist()
     


# In[79]:


data2.head(2)


# In[80]:


#gets the closest data to the cities data

import pandas as pd
from scipy.spatial.distance import cdist

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]

#cities is in df
#data is in data

df['point'] = [(x, y) for x,y in zip(df['Latitude'], df['Longitude'])]

data['point'] = [(x, y) for x,y in zip(data['lat_bnds'], data['lon_bnds'])]
data1['point1'] = [(x, y) for x,y in zip(data1['lat_bnds'], data1['lon_bnds'])]
data2['point2'] = [(x, y) for x,y in zip(data2['lat_bnds'], data2['lon_bnds'])]

col1T = col + " " + date_old
col2T = col1 + " " + date_old
col3T = col1+ " " + date_new

df['closest'] = [closest_point(x, list(data['point'])) for x in df['point']]
df[col1T] = [match_value(data, 'point', x, col) for x in df['closest']]

df['closest1'] = [closest_point(x, list(data1['point1'])) for x in df['point']]
df[col2T] = [match_value(data1, 'point1', x, col1) for x in df['closest1']]

df['closest2'] = [closest_point(x, list(data2['point2'])) for x in df['point']]
df[col3T] = [match_value(data2, 'point2', x, col1) for x in df['closest2']]


# In[81]:


df['Region'] = 'blank'

df.loc[df['City'] == 'Andorra', 'Region'] = 'AND'
df.loc[df['City'] == 'Angola', 'Region'] = 'AGO'
df.loc[df['City'] == 'Argentina', 'Region'] = 'ARG'
df.loc[df['City'] == 'Armenia', 'Region'] = 'ARM'
df.loc[df['City'] == 'Aruba', 'Region'] = 'ABW'
df.loc[df['City'] == 'Australia', 'Region'] = 'AUS'
df.loc[df['City'] == 'Austria', 'Region'] = 'AUT'
df.loc[df['City'] == 'Azerbaijan', 'Region'] = 'AZE'
df.loc[df['City'] == 'Bahrain', 'Region'] = 'BHR'
df.loc[df['City'] == 'Bangladesh', 'Region'] = 'BGD'
df.loc[df['City'] == 'Belarus', 'Region'] = 'BLR'
df.loc[df['City'] == 'Belgium', 'Region'] = 'BEL'
df.loc[df['City'] == 'Benin', 'Region'] = 'BEN'
#N/A
df.loc[df['City'] == 'Brazil', 'Region'] = 'BRA'
df.loc[df['City'] == 'Bulgaria', 'Region'] = 'BGR'
df.loc[df['City'] == 'Cameroon', 'Region'] = 'CMR'
df.loc[df['City'] == 'Canada', 'Region'] = 'CAN'
df.loc[df['City'] == 'Cabo Verde', 'Region'] = 'CPV'
df.loc[df['City'] == 'Chile', 'Region'] = 'CHL'
df.loc[df['City'] == 'China', 'Region'] = 'CHN'
df.loc[df['City'] == 'Colombia', 'Region'] = 'COL'
df.loc[df['City'] == 'Congo, Republic of', 'Region'] = 'COG'
df.loc[df['City'] == 'Costa Rica', 'Region'] = 'CRI'
df.loc[df['City'] == 'Croatia', 'Region'] = 'HRV'
df.loc[df['City'] == 'Cyprus', 'Region'] = 'CYP'
#N/A
df.loc[df['City'] == 'Denmark', 'Region'] = 'DNK'
df.loc[df['City'] == 'Dominican Republic', 'Region'] = 'DOM'
df.loc[df['City'] == 'Ecuador', 'Region'] = 'ECU'
df.loc[df['City'] == 'Egypt', 'Region'] = 'EGY'
df.loc[df['City'] == 'El Salvador', 'Region'] = 'SLV'
df.loc[df['City'] == 'Estonia', 'Region'] = 'EST'
df.loc[df['City'] == 'Ethiopia', 'Region'] = 'ETH'
df.loc[df['City'] == 'Finland', 'Region'] = 'FIN'
df.loc[df['City'] == 'France', 'Region'] = 'FRA'
df.loc[df['City'] == 'Gabon', 'Region'] = 'GAB'
df.loc[df['City'] == 'Georgia', 'Region'] = 'GEO'
df.loc[df['City'] == 'Germany', 'Region'] = 'DEU'
df.loc[df['City'] == 'Ghana', 'Region'] = 'GHA'
df.loc[df['City'] == 'Greece', 'Region'] = 'GRC'
df.loc[df['City'] == 'Guatemala', 'Region'] = 'GTM'
df.loc[df['City'] == 'Hong Kong', 'Region'] = 'HKG'
df.loc[df['City'] == 'Hungary', 'Region'] = 'HUN'
df.loc[df['City'] == 'Iceland', 'Region'] = 'ISL'
df.loc[df['City'] == 'India', 'Region'] = 'IND'
df.loc[df['City'] == 'Indonesia', 'Region'] = 'IDN'
df.loc[df['City'] == 'Iraq', 'Region'] = 'IRQ'
df.loc[df['City'] == 'Ireland', 'Region'] = 'IRL'
df.loc[df['City'] == 'Israel', 'Region'] = 'ISR'
df.loc[df['City'] == 'Italy', 'Region'] = 'ITA'
df.loc[df['City'] == 'Jamaica', 'Region'] = 'JAM'
df.loc[df['City'] == 'Japan', 'Region'] = 'JPN'
#N/A
df.loc[df['City'] == 'Kazakhstan', 'Region'] = 'KAZ'
df.loc[df['City'] == 'Kenya', 'Region'] = 'KEN'
df.loc[df['City'] == 'Korea', 'Region'] = 'KOR'
df.loc[df['City'] == 'Kuwait', 'Region'] = 'KWT'
df.loc[df['City'] == 'Laos', 'Region'] = 'JOR'
df.loc[df['City'] == 'Latvia', 'Region'] = 'LVA'
df.loc[df['City'] == 'Lebanon', 'Region'] = 'LBN'
df.loc[df['City'] == 'Lesotho', 'Region'] = 'LSO'
df.loc[df['City'] == 'Lithuania', 'Region'] = 'LTU'
df.loc[df['City'] == 'Luxembourg', 'Region'] = 'LUX'
df.loc[df['City'] == 'Macao', 'Region'] = 'MAC'
df.loc[df['City'] == 'Malaysia', 'Region'] = 'MYS'
df.loc[df['City'] == 'Maldives', 'Region'] = 'MDV'
df.loc[df['City'] == 'Malta', 'Region'] = 'MLT'
df.loc[df['City'] == 'Mexico', 'Region'] = 'MEX'
df.loc[df['City'] == 'Mongolia', 'Region'] = 'MNG'
df.loc[df['City'] == 'Morocco', 'Region'] = 'MAR'
df.loc[df['City'] == 'Mozambique', 'Region'] = 'MOZ'
df.loc[df['City'] == 'Namibia', 'Region'] = 'NAM'
df.loc[df['City'] == 'Netherlands', 'Region'] = 'NLD'
df.loc[df['City'] == 'New Zealand', 'Region'] = 'NZL'
df.loc[df['City'] == 'Nicaragua', 'Region'] = 'NIC'
df.loc[df['City'] == 'Nigeria', 'Region'] = 'NGA'
df.loc[df['City'] == 'North Macedonia', 'Region'] = 'MKD'
df.loc[df['City'] == 'Norway', 'Region'] = 'NOR'
df.loc[df['City'] == 'Oman', 'Region'] = 'OMN'
df.loc[df['City'] == 'Pakistan', 'Region'] = 'PAK'
df.loc[df['City'] == 'Panama', 'Region'] = 'PAN'
df.loc[df['City'] == 'Paraguay', 'Region'] = 'PRY'
df.loc[df['City'] == 'Peru', 'Region'] = 'PER'
df.loc[df['City'] == 'Philippines', 'Region'] = 'PHL'
df.loc[df['City'] == 'Poland', 'Region'] = 'POL'
df.loc[df['City'] == 'Portugal', 'Region'] = 'PRT'
df.loc[df['City'] == 'Qatar', 'Region'] = 'QAT'
#N/A
df.loc[df['City'] == 'Romania', 'Region'] = 'ROU'
df.loc[df['City'] == 'Russia', 'Region'] = 'RUS'
df.loc[df['City'] == 'Rwanda', 'Region'] = 'RWA'
df.loc[df['City'] == 'San Marino', 'Region'] = 'SMR'
df.loc[df['City'] == 'Saudi Arabia', 'Region'] = 'SAU'
df.loc[df['City'] == 'Serbia', 'Region'] = 'SRB'
df.loc[df['City'] == 'Seychelles', 'Region'] = 'SYC'
df.loc[df['City'] == 'Singapore', 'Region'] = 'SGP'
df.loc[df['City'] == 'Slovakia', 'Region'] = 'SVK'
df.loc[df['City'] == 'Slovenia', 'Region'] = 'SVN'
df.loc[df['City'] == 'South Africa', 'Region'] = 'ZAF'
df.loc[df['City'] == 'Spain', 'Region'] = 'ESP'
df.loc[df['City'] == 'Sri Lanka', 'Region'] = 'LKA'
df.loc[df['City'] == 'Suriname', 'Region'] = 'SUR'
df.loc[df['City'] == 'Sweden', 'Region'] = 'SWE'
df.loc[df['City'] == 'Switzerland', 'Region'] = 'CHE'
df.loc[df['City'] == 'Taiwan', 'Region'] = 'TWN'
df.loc[df['City'] == 'Thailand', 'Region'] = 'THA'
df.loc[df['City'] == 'Tunisia', 'Region'] = 'TUN'
df.loc[df['City'] == 'Turkey', 'Region'] = 'TUR'
df.loc[df['City'] == 'Uganda', 'Region'] = 'UGA'
df.loc[df['City'] == 'Ukraine', 'Region'] = 'UKR'
df.loc[df['City'] == 'United Kingdom', 'Region'] = 'GBR'
df.loc[df['City'] == 'United States of America', 'Region'] = 'USA'
df.loc[df['City'] == 'Uruguay', 'Region'] = 'URY'
df.loc[df['City'] == 'Uzbekistan', 'Region'] = 'UZB'
df.loc[df['City'] == 'Vietnam', 'Region'] = 'VNM'
df.loc[df['City'] == 'Zambia', 'Region'] = 'ZMB'


# In[82]:



df


# In[83]:


import plotly.graph_objects as go
import pandas as pd

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

# options
# col1T = col + " " + date_old
# col2T = col1 + " " + date_old
# col3T = col1+ " " + date_new



fig = go.Figure(data=go.Choropleth(
    locations = df['Region'],
    z = df[col1T],
    text = df['City'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='dark grey',
    marker_line_width=0.5,
    #colorbar_tickprefix = '$',
    colorbar_title = 'Days',
))

fig.update_layout(
    title_text=col1T,
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.05,
        xref='paper',
        yref='paper',
        text=('Climate Data Factory'
              ' Model GFDL-ESM2M RCP85'),
        showarrow = False
    )]

)

fig.show()


# In[84]:


#lon, lat = -86.276, 30.935 
#zoom_start = 5

m = folium.Map(location=[0,0],
              
               tiles='CartoDB positron',
               zoom_start=1.5,
               control_scale=True,
               max_bounds = True,
               zoom_control = True,
               max_zoom =6,
               min_zoom =2
              )
gradient_p = {0.4: 'white', 0.65: 'transparent',0.8: 'purple'}
gradient = {0.4: 'white', 0.65: 'transparent',0.8: 'blue'}
gradient_r = {0.4: 'white', 0.65: 'transparent',0.8: 'red'}
gradient_b = {0.4: 'white', 0.65: 'transparent',0.8: 'blue'}
#add in Heatmap data\n",

HeatMap(data_value,min_opacity=0.1,gradient=gradient_p).add_to(
    folium.FeatureGroup(name= col1T).add_to(m))
HeatMap(data_value1,min_opacity=0.1,gradient=gradient_b).add_to(
    folium.FeatureGroup(name= col2T).add_to(m))
HeatMap(data_value2,min_opacity=0.1,gradient=gradient_r).add_to(
    folium.FeatureGroup(name= col3T).add_to(m))

                                                             
                                                             
feature_group = FeatureGroup(name="Cities " + col1T )
#add the locations\n",
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point],popup=df['City'][point] + " " + str(round(df[col1T][point],2) ),icon=folium.map.Icon(color='purple')).add_to(folium.FeatureGroup(name= "data")).add_to(feature_group)
#    folium.Marker(locationlist[point], popup=df['City'][point] + " " + str(round(df[col1][point],2) )).add_to(m) 
feature_group.add_to(m)

feature_group = FeatureGroup(name="Cities " + col2T)
#add the locations\n",
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point],
                  popup=df['City'][point] + " " + str(round(df[col2T][point],2) ),icon=folium.map.Icon(color='blue')).add_to(folium.FeatureGroup(name= "data")).add_to(feature_group)
#    folium.Marker(locationlist[point], popup=df['City'][point] + " " + str(round(df[col1][point],2) )).add_to(m) 
feature_group.add_to(m)

feature_group = FeatureGroup(name="Cities " + col3T )
#add the locations\n",
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point],
                  popup=df['City'][point] + " " + str(round(df[col3T][point],2) ),icon=folium.map.Icon(color='red')).add_to(folium.FeatureGroup(name= "data")).add_to(feature_group)
#    folium.Marker(locationlist[point], popup=df['City'][point] + " " + str(round(df[col1][point],2) )).add_to(m) 
feature_group.add_to(m)


folium.LayerControl().add_to(m)
m


# In[86]:


df_tab = df.drop(['point', 'closest1','closest','closest2'], axis = 1)
df_tab


# In[ ]:




