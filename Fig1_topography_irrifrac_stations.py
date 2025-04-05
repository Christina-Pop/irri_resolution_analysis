#!/usr/bin/env python
# coding: utf-8

# Create Figure 1 

import os 
import pandas as pd 
import glob
import csv
import xarray as xr
import numpy as np
from datetime import datetime
from datetime import date
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt

from functions_correcting_time import correct_timedim
from functions_plotting import plot_rotvar, plot_rotvar_adjust_cbar, rotated_pole, plot_frequency_histogram, plot_density_histogram
from functions_reading_files import read_efiles, read_efiles_in_chunks
from functions_rotation import rotated_coord_transform, np_lon, np_lat
from functions_idw import idw_for_model 


# create plotting directory 
dir_working = os.getcwd()
# creates dir in parent directory
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

############################### read REMO data ####################################################################

# read data - static var from not irrigated sim
year = 2017
month = 0
#
exp_number_noirri_SGAR = "067108"
exp_number_noirri_GAR = "067026"
#
varlist_SGAR = [ "FIB", "IRRIFRAC"]
var_num_list_SGAR = ["129", "743"]
#
varlist_GAR = [ "FIB", "IRRIFRAC"]
var_num_list_GAR = ["129", "743"]

# adapt your paths where the data is stored.
# Please also check the functions in functions_reading_files.py. 
data_path_SGAR_noirri = "/noirri0275"
data_path_GAR_noirri = "/noirri11"

# SGAR 
for var_SGAR, var_num_SGAR in zip(varlist_SGAR, var_num_list_SGAR):
    print('reading ', var_SGAR)
    single_var_data_SGAR_noirri = read_efiles_in_chunks(data_path_SGAR_noirri, var_SGAR, var_num_SGAR, exp_number_noirri_SGAR, year, month, 5, 10, None, None)
   
    if var_SGAR == varlist_SGAR[0]:
        ds_var_noirri_SGAR = single_var_data_SGAR_noirri
    else:
        ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, single_var_data_SGAR_noirri])
# GAR
for var_GAR, var_num_GAR in zip(varlist_GAR, var_num_list_GAR):
    print('reading ', var_GAR)
    single_var_data_GAR_noirri = read_efiles(data_path_GAR_noirri, var_GAR, var_num_GAR, exp_number_noirri_GAR, year, month)
   
    if var_GAR == varlist_GAR[0]:
        ds_var_noirri_GAR = single_var_data_GAR_noirri
    else:
        ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, single_var_data_GAR_noirri])

dsnoirr_newtime_SGAR = correct_timedim(ds_var_noirri_SGAR)
dsnoirr_newtime_GAR = correct_timedim(ds_var_noirri_GAR)

# cut out from GAR and SGAR the same region 
def cut_same_area(source_area, target_area):
    min_rlon_target=target_area.rlon[0].values 
    max_rlon_target=target_area.rlon[-1].values

    min_rlat_target=target_area.rlat[0].values 
    max_rlat_target=target_area.rlat[-1].values
    
    cutted_ds = source_area.sel(rlon=slice(min_rlon_target,max_rlon_target), \
                                rlat=slice(min_rlat_target,max_rlat_target))
    return cutted_ds

dsnoirr_newtime_GAR_cut = cut_same_area(dsnoirr_newtime_GAR, dsnoirr_newtime_SGAR)

# cut out the Po valley 
po_noirri_GAR  = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_SGAR = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR)


##################################### functions for topography ##############################################

def earth_feature(name_obj, res, col, cat):
    lines = cfeature.NaturalEarthFeature(
        category=cat, name=name_obj, scale=res, facecolor=col
    )
    return lines


def earth_feature_area(name_obj, res, cat, col):
    area = cfeature.NaturalEarthFeature(
        category=cat, name=name_obj, scale=res, facecolor=col
    )
    return area


states = earth_feature("admin_1_states_provinces_lines", "10m", "none", "cultural")
land = earth_feature_area("land", "10m", "physical", "none")
rivers = earth_feature("rivers_lake_centerlines", "10m", "none", "physical")
coastline = earth_feature("coastline", "10m", "none", "physical")
borders = earth_feature("admin_0_boundary_lines_land", "10m", "none", "cultural")
ocean = earth_feature_area("ocean", "10m", "physical", "none")


def plot_topography(fig, topo, axs, caxs, unit, label, levels, ticks, cbar_orient): 
    cmap_custom = mpl.colors.LinearSegmentedColormap.from_list('custom', 
                                             [#(0,    'lightblue'),
                                              (0.0, 'green'),
                                              (0.25, 'khaki'),
                                              (0.5, 'goldenrod'),
                                              (0.9,    'saddlebrown'),
                                              (1.,'white')], N=200)




    axs.add_feature(cartopy.feature.OCEAN,  zorder=3, edgecolor='k', linewidth=0.5)
    tplot = plot_rotvar(fig = fig, varvalue = topo, axs=axs, cbaxes = caxs, unit=unit, label = label, 
                        cmap=plt.matplotlib.colormaps.get_cmap(cmap_custom), levels = levels, ticks = ticks, extend_scale = 'max', cbar_orient = cbar_orient)
    axs.gridlines(linewidth=0.7, color='gray', alpha=0.8, linestyle='--',zorder=3)
    cbar = fig.colorbar(tplot, cax=caxs, orientation=cbar_orient, label=label, ticks=ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.outline.set_visible(True)
    return tplot 


################################# Precipitation ###############################################


# READ STATION DATA 
# read data for EMILIA-ROMAGNA
# 1. extract names of stations
path_ER = "/work/ch0636/g300099/data/station/EMILIA_ROMAGNA/precipitation"
sensor_files_ER = glob.glob(os.path.join(path_ER, "*.csv"))
import csv

stations_df = pd.DataFrame()
for single_file in sensor_files_ER:

    with open(single_file) as file_obj:

        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            try:
                if len(row) == 10:
                    if ((row[0] == "Nome della stazione") & (len(stations_df) == 0)): 
                        stations_df = pd.DataFrame(columns = row)
                    elif ((row[0] == "Nome della stazione") & (len(stations_df) > 0)): 
                        pass
                    else:
                        stations_df.loc[len(stations_df)] = row
            except:
                continue
station_list = stations_df['Nome della stazione'].values


# 2. read the data 
path_ER = "/work/ch0636/g300099/data/station/EMILIA_ROMAGNA/precipitation"
sensor_files_ER = glob.glob(os.path.join(path_ER, "*.csv"))

# constants
number_of_ts = 5142-4 # (lines in table)
station_all_data_ER = pd.DataFrame()

for sensor_file in sensor_files_ER: 
    print('file:', sensor_file)
    counter = 0
    readable_lines = []
    station_list_file =[]
    sensor_df = pd.DataFrame()

    # find lines to read 
    with open(sensor_file) as file_obj:
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                counter = counter+1
                try:
                    if ((row[0] in station_list) & (len(row)) == 1):
                        station_list_file.append(row[0])
                        readable_lines.append(counter)
                    else:
                        pass
                except:
                    continue
            #print(station_list_file)
        
    # calculate number of rows 
    data_length_list = list(np.diff(readable_lines)-3)
    data_length_list.append(0)
    
    # read only specific rows 
    for data_length, station_name, skipline in zip(data_length_list, station_list_file, readable_lines):
        #print('reading now:', station_name, ' from line: ', skipline, ' to line: ', skipline+data_length )
        if skipline == readable_lines[-1]:
            sensor_data = pd.read_csv(sensor_file, skiprows=skipline, skipfooter = 10, engine='python' )
            sensor_data['sensor_name'] = [station_name]*len(sensor_data)
        else: 
            sensor_data = pd.read_csv(sensor_file, skiprows=skipline, nrows = data_length )
            sensor_data['sensor_name'] = [station_name]*len(sensor_data)

        if len(sensor_data) == 0: 
           # print('Station ', station_name, 'has no data. Please check!')
            pass 
        else:
            # adjust the df: dropping, renaming, datetime_index, adding station
            sensor_data = sensor_data.drop(columns = ['Inizio validità (UTC)'])\
                .rename(columns={"Fine validità (UTC)": "Date","Precipitazione cumulata su 1 ora (KG/M**2)":"PRECIP_station"})
       
            sensor_data['Date'] = (pd.to_datetime(sensor_data['Date']))
            sensor_data = sensor_data.set_index('Date')
            sensor_data = sensor_data["2017-02-28 23:00:00+00:00":"2017-09-01 00:00:00+00:00"] 
            ## add lon & lat 
            sensor_data['lon'] = [stations_df['Longitudine (Gradi Centesimali)'][stations_df['Nome della stazione'] == station_name ].values[0]] * len(sensor_data)
            sensor_data['lat'] = [stations_df['Latitudine (Gradi Centesimali)'][stations_df['Nome della stazione'] == station_name ].values[0]] * len(sensor_data)
            
            #if len(sensor_data) != 4393: 
              #  print('Station ', station_name, 'has ',4393-len(sensor_data) ,' missing timesteps. Please check!')
            ## concat df from one file to one df 
            sensor_df = pd.concat([sensor_df, sensor_data])
        station_all_data_ER = pd.concat([station_all_data_ER, sensor_df])
# EMILIA-ROMAGNA is in UTC, therefore converted to CET 
# convert UTC to CET and remove tzinfo 
station_all_data_ER.index = station_all_data_ER.index.tz_convert(tz='CET')
station_all_data_ER.index = station_all_data_ER.index.tz_localize(None)
station_all_data_ER = station_all_data_ER.sort_index()
station_all_data_ER = station_all_data_ER["2017-03-01 00:00:00":"2017-08-31 23:50:00"] 
station_all_data_ER['lon']= station_all_data_ER.lon.astype(float)
station_all_data_ER['lat']= station_all_data_ER.lat.astype(float)

# rotate EM data
stations_rot  = []
rot_longitude = []
rot_latitude  = []
for i, (lon, lat) in enumerate(zip(station_all_data_ER.lon, station_all_data_ER.lat)): 
    stations_rot.append(rotated_coord_transform( lon, lat, 
    np_lon, np_lat, direction="geo2rot") )
    rot_longitude.append(round(stations_rot[i][0], 7))
    rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_ER["rlon"] = rot_longitude
station_all_data_ER["rlat"] = rot_latitude
station_all_data_ER['region'] = ['Emilia-Romagna']*len(station_all_data_ER)
# 1. drop weird values 
station_all_data_ER.loc[((station_all_data_ER.PRECIP_station < 0) | (station_all_data_ER.PRECIP_station  > 100 )), "PRECIP_station" ] = np.nan
# 2. selsct time period 
station_all_data_ER = station_all_data_ER['2017-03-01 00:00:00':'2017-07-31 23:00:00']
# 3. drop duplicates 
station_all_data_ER.drop_duplicates()
station_all_data_ER = station_all_data_ER.drop(columns = 'sensor_name')
# 4. set time 
station_all_data_ER['time'] = station_all_data_ER.index
# we will drop nan after we combine REMO & ER data 
station_all_data_ER

# LOMBARDIA 

# read data Lombardia 
# 1. export relevant sensors for 1. variable, 2. time range 
var_name_it = "Precipitazione"

# read station data 
path = "/work/ch0636/g300099/data/station/LOMBARDIA/"
station_details_file = "station_data/Stazioni_Meteorologiche_20231025.csv"
station_data = pd.read_csv(os.path.join(path, station_details_file), delimiter=",")

date_format = '%d/%m/%Y'
date_start = datetime(2017, 3, 1)
date_stop = datetime(2017, 8, 1)

station_data['DataStart'] = pd.to_datetime(station_data["DataStart"], format = date_format) 
station_data['DataStop'] = pd.to_datetime(station_data["DataStop"], format = date_format) 

var_stations = station_data[((station_data["Tipologia"]==var_name_it) & \
                            (station_data["DataStart"]<=date_start)  & \
                            ((station_data["DataStop"]>=date_stop)    |\
                            (pd.isna(station_data["DataStop"]))))  ] 
# 2. read sensor data 
# LOMBARDIA is in "sun hours" --> LT 
date_format = '%d/%m/%Y %H:%M:%S'

path = "/work/ch0636/g300099/data/station/LOMBARDIA/"
sensor_file = "data/2017.csv"
sensor_data = pd.read_csv(os.path.join(path, sensor_file), delimiter=",")
sensor_data["time"] = pd.to_datetime(sensor_data["Data"], format = date_format) 
sensor_data = sensor_data.set_index(sensor_data["time"])
#sensor_data = sensor_data.replace({-9999: np.nan})

station_all_data_LOM = pd.DataFrame()
# combine var values with sensor id, lon, lat and aggregate to hourly values
for sensor_id in var_stations["IdSensore"]: 
    #print('sensor id:', sensor_id)
    sensor_df = sensor_data[sensor_data["IdSensore"]==sensor_id][['Valore']].sort_index()
    # for precipitation hourly sum
    if var_name_it == "Precipitazione":
        #sensor_df_clean = 
        sensor_df_mean = sensor_df.resample("1H").sum(min_count=6)
    # for temperature hourly mean
    elif var_name_it == "Temperatura": 
        sensor_df_mean = sensor_df.resample("1H").mean(min_count=6)
    else: 
        print('var_name_it '+str(var_name_it)+' unknown!')
    sensor_df_mean['IdSensore'] = [sensor_id]*len(sensor_df_mean)
    lon, lat = station_data[station_data["IdSensore"]==sensor_id][['lng','lat']].values[0]
    sensor_df_mean['lon'] = [lon]*len(sensor_df_mean)
    sensor_df_mean['lat'] = [lat]*len(sensor_df_mean)
    # combine all sensor to a big df 
    station_all_data_LOM = pd.concat([station_all_data_LOM, sensor_df_mean])
#remove duplicates completely! in our case: 2377 and 2385 
station_all_data_LOM.rename(columns = {"Valore":"PRECIP_station"}, inplace = True)
#1. take out weird values 
station_all_data_LOM.loc[((station_all_data_LOM.PRECIP_station < 0) | (station_all_data_LOM.PRECIP_station  > 100 )), "PRECIP_station" ] = np.nan
station_all_data_LOM = station_all_data_LOM[(station_all_data_LOM['IdSensore'] != 2377) & (station_all_data_LOM['IdSensore'] != 2385) & (station_all_data_LOM['IdSensore'] != 9106)]
# 2. set time range 
station_all_data_LOM = station_all_data_LOM["2017-03-01 00:00:00":"2017-07-31 23:00:00"]
# 3. drop dplicates 
station_all_data_LOM.drop_duplicates()
# 4. set time 
station_all_data_LOM['time'] = station_all_data_LOM.index
station_all_data_LOM
# 
# rotate
stations_rot = []
rot_longitude = []
rot_latitude = []
for i, (lon, lat) in enumerate(zip(station_all_data_LOM.lon, station_all_data_LOM.lat)): 
    stations_rot.append(rotated_coord_transform( lon, lat, 
    np_lon, np_lat, direction="geo2rot") )
    rot_longitude.append(round(stations_rot[i][0], 7))
    rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_LOM["rlon"] = rot_longitude
station_all_data_LOM["rlat"] = rot_latitude
station_all_data_LOM['region'] = ['Lombardia']*len(station_all_data_LOM)
station_all_data_LOM = station_all_data_LOM.drop(columns = 'IdSensore')
station_all_data_LOM
 
# VENTEO
# read data from Veneto 
import glob

station_path_VEN = '/work/ch0636/g300099/data/station/VENETO'
station_files_VEN = glob.glob(os.path.join(station_path_VEN, '*Precipitazione*.txt'))
station_data_VEN = pd.concat((pd.read_csv(f, sep = ';') for f in station_files_VEN), ignore_index=True)
station_data_VEN['time'] = pd.to_datetime(station_data_VEN['DATA'].astype(str)  + ' ' +station_data_VEN['ORA'].astype(str))
station_data_VEN = station_data_VEN.rename(columns={'STAZIONE':'name','VALORE':'PRECIP_station'}).drop(columns=(['DATA','ORA','SENSORE']))

# 
station_detail_file_VEN = "Meteological Stations ARPA VENETO  - Dott.ssa ASMUS.xlsx"
station_detail = pd.read_excel(os.path.join(station_path_VEN, station_detail_file_VEN ),sheet_name = 'Foglio2',skiprows = 11,decimal='.' )
station_detail = station_detail.rename(columns={'Unnamed: 0':'Code','Unnamed: 1':'name','dal':'start date','Unnamed: 3':'stop date',
                                               'Unnamed: 4':'type','Unnamed: 5':'quota metri','Unnamed: 6':'commune', 
                                               'Unnamed: 7':'Gauss_x','Unnamed: 8':'Gauss_y','Unnamed: 9':'geo_lat','Unnamed: 10':'geo_lon', 
                                               'Unnamed: 11':'lat_deg','Unnamed: 12':'lat_min','Unnamed: 13':'lat_sec','Unnamed: 14':'lon_deg',
                                               'Unnamed: 15':'lon_min','Unnamed: 16':'lon_sec','Unnamed: 17':'basin','Unnamed: 18':'sotto basin'
                                               })
station_detail = station_detail.drop(columns=(['Code','type', 'quota metri','commune','Gauss_x','Gauss_y','geo_lat','geo_lon','basin','sotto basin']))
station_detail['lat_sec'] = station_detail['lat_sec']/1000
station_detail['lon_sec'] = station_detail['lon_sec']/1000

degrees = station_detail['lat_deg']
minutes = station_detail['lat_min']
seconds = station_detail['lat_sec']
station_detail['lat']=(degrees + ((minutes/60) + seconds/(60*60)))
degrees = station_detail['lon_deg']
minutes = station_detail['lon_min']
seconds = station_detail['lon_sec']
station_detail['lon']=(degrees + ((minutes/60) + seconds/(60*60)))
station_detail = station_detail.drop(columns=(['lat_deg','lat_min','lat_sec','lon_deg','lon_min','lon_sec']))

# 
station_all_data_VEN = station_data_VEN.merge(station_detail, how='left', on=['name'])
station_all_data_VEN = station_all_data_VEN.set_index('time', drop=True)
station_all_data_VEN = station_all_data_VEN['2017-03-01 00:00:00':'2017-07-31 23:00:00']
station_all_data_VEN
# 
# rotate
stations_rot = []
rot_longitude = []
rot_latitude = []
for i, (lon, lat) in enumerate(zip(station_all_data_VEN.lon, station_all_data_VEN.lat)): 
    stations_rot.append(rotated_coord_transform( lon, lat, 
    np_lon, np_lat, direction="geo2rot") )
    rot_longitude.append(round(stations_rot[i][0], 7))
    rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_VEN["rlon"] = rot_longitude
station_all_data_VEN["rlat"] = rot_latitude
station_all_data_VEN['region'] = ['Veneto']*len(station_all_data_VEN)
station_all_data_VEN = station_all_data_VEN.drop(columns=(['name','start date','stop date']))
station_all_data_VEN['time'] = station_all_data_VEN.index
station_all_data_VEN


# merge ER, LOM, and VEN 
df_all_station = pd.concat([station_all_data_ER, station_all_data_LOM, station_all_data_VEN]) 
print('expected length', (len(station_all_data_ER) + len(station_all_data_LOM)+len(station_all_data_VEN)))
df_all_station_clean = df_all_station.drop_duplicates()
df_all_station_clean

# clean stations outside from EVAL area 

df_all_station_clean_in = df_all_station_clean.where((df_all_station_clean.rlon>sorted(po_noirri_SGAR.rlon.values)[0])
                                                     &(df_all_station_clean.rlon<sorted(po_noirri_SGAR.rlat.values)[-1])
                                                     &(df_all_station_clean.rlat>sorted(po_noirri_SGAR.rlat.values)[0])
                                                     &(df_all_station_clean.rlat<sorted(po_noirri_SGAR.rlat.values)[-1]))
df_all_station_precip = df_all_station_clean_in


############################ Temperature ################################

# TEMPERATURE
# READ STATION DATA 
# read data for EMILIA-ROMAGNA
# 1. extract names of stations
path_ER = "/work/ch0636/g300099/data/station/EMILIA_ROMAGNA/temperature"
sensor_files_ER = glob.glob(os.path.join(path_ER, "*.csv"))
import csv

stations_df = pd.DataFrame()
for single_file in sensor_files_ER:

    with open(single_file) as file_obj:

        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            try:
                if len(row) == 10:
                    if ((row[0] == "Nome della stazione") & (len(stations_df) == 0)): 
                        stations_df = pd.DataFrame(columns = row)
                    elif ((row[0] == "Nome della stazione") & (len(stations_df) > 0)): 
                        pass
                    else:
                        stations_df.loc[len(stations_df)] = row
            except:
                continue
station_list = stations_df['Nome della stazione'].values

# 2. read the data 
path_ER = "/work/ch0636/g300099/data/station/EMILIA_ROMAGNA/temperature"
sensor_files_ER = glob.glob(os.path.join(path_ER, "*.csv"))

# constants
#number_of_ts = 5142-4
station_all_data_ER = pd.DataFrame()

for sensor_file in sensor_files_ER: 
    print('file:', sensor_file)
    counter = 0
    readable_lines = []
    station_list_file =[]
    sensor_df = pd.DataFrame()

    # find lines to read 
    with open(sensor_file) as file_obj:
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                counter = counter+1
                try:
                    if ((row[0] in station_list) & (len(row)) == 1):
                        station_list_file.append(row[0])
                        readable_lines.append(counter)
                    else:
                        pass
                except:
                    continue
            #print(station_list_file)
        
    # calculate number of rows 
    data_length_list = list(np.diff(readable_lines)-3)
    data_length_list.append(0)
    
    # read only specific rows 
    for data_length, station_name, skipline in zip(data_length_list, station_list_file, readable_lines):
        #print('reading now:', station_name, ' from line: ', skipline, ' to line: ', skipline+data_length )
        if skipline == readable_lines[-1]:
            sensor_data = pd.read_csv(sensor_file, skiprows=skipline, skipfooter = 11, engine='python' )
            sensor_data['sensor_name'] = [station_name]*len(sensor_data)
        else: 
            sensor_data = pd.read_csv(sensor_file, skiprows=skipline, nrows = data_length )
            sensor_data['sensor_name'] = [station_name]*len(sensor_data)

        if len(sensor_data) == 0: 
           # print('Station ', station_name, 'has no data. Please check!')
            pass 
        else:
            # adjust the df: dropping, renaming, datetime_index, adding station
            sensor_data = sensor_data.drop(columns = ['Inizio validità (UTC)'])\
                .rename(columns={"Fine validità (UTC)": "Date","Temperatura dell'aria media oraria a 2 m dal suolo (°C)":"TEMP_station"})
       
            sensor_data['Date'] = (pd.to_datetime(sensor_data['Date']))
            sensor_data = sensor_data.set_index('Date')
            sensor_data = sensor_data["2017-02-28 23:00:00+00:00":"2017-09-01 00:00:00+00:00"] 
            ## add lon & lat 
            sensor_data['lon'] = [stations_df['Longitudine (Gradi Centesimali)'][stations_df['Nome della stazione'] == station_name ].values[0]] * len(sensor_data)
            sensor_data['lat'] = [stations_df['Latitudine (Gradi Centesimali)'][stations_df['Nome della stazione'] == station_name ].values[0]] * len(sensor_data)
            
            #if len(sensor_data) != 4393: 
              #  print('Station ', station_name, 'has ',4393-len(sensor_data) ,' missing timesteps. Please check!')
            ## concat df from one file to one df 
            sensor_df = pd.concat([sensor_df, sensor_data])
        station_all_data_ER = pd.concat([station_all_data_ER, sensor_df])
station_all_data_ER = station_all_data_ER.sort_index()
station_all_data_ER.index = station_all_data_ER.index.tz_convert(tz='CET')
station_all_data_ER.index = station_all_data_ER.index.tz_localize(None)
station_all_data_ER["2017-03-01 00:00:00":"2017-08-31 23:00:00"] 
station_all_data_ER['lon']= station_all_data_ER.lon.astype(float)
station_all_data_ER['lat']= station_all_data_ER.lat.astype(float)
# rotate EM data
stations_rot  = []
rot_longitude = []
rot_latitude  = []
for i, (lon, lat) in enumerate(zip(station_all_data_ER.lon, station_all_data_ER.lat)): 
    stations_rot.append(rotated_coord_transform( lon, lat, 
    np_lon, np_lat, direction="geo2rot") )
    rot_longitude.append(round(stations_rot[i][0], 7))
    rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_ER["rlon"] = rot_longitude
station_all_data_ER["rlat"] = rot_latitude
station_all_data_ER['region'] = ['Emilia-Romagna']*len(station_all_data_ER)
# 1. drop weird values 
station_all_data_ER.loc[((station_all_data_ER.TEMP_station  > 100 )), "TEMP_station" ] = np.nan
# 2. selsct time period 
station_all_data_ER = station_all_data_ER['2017-03-01 00:00:00':'2017-07-31 23:00:00']
# 3. drop duplicates 
station_all_data_ER.drop_duplicates()
station_all_data_ER = station_all_data_ER.drop(columns = 'sensor_name')
# 4. set time 
station_all_data_ER['time'] = station_all_data_ER.index
# we will drop nan after we combine REMO & ER data 
station_all_data_ER

# LOMBARDIA 

# read data Lombardia 
# 1. export relevant sensors for 1. variable, 2. time range 
var_name_it = "Temperatura"

# read station data 
path = "/work/ch0636/g300099/data/station/LOMBARDIA/"
station_details_file = "station_data/Stazioni_Meteorologiche_20231025.csv"
station_data = pd.read_csv(os.path.join(path, station_details_file), delimiter=",")

date_format = '%d/%m/%Y'
date_start = datetime(2017, 3, 1)
date_stop = datetime(2017, 10, 1)

station_data['DataStart'] = pd.to_datetime(station_data["DataStart"], format = date_format) 
station_data['DataStop'] = pd.to_datetime(station_data["DataStop"], format = date_format) 

var_stations = station_data[((station_data["Tipologia"]==var_name_it) & \
                            (station_data["DataStart"]<=date_start)  & \
                            ((station_data["DataStop"]>=date_stop)    |\
                            (pd.isna(station_data["DataStop"]))))  ] 
# 2. read sensor data 
# LOMBARDIA is in "sun hours" --> LT 
date_format = '%d/%m/%Y %H:%M:%S'

path = "/work/ch0636/g300099/data/station/LOMBARDIA/"
sensor_file = "data/2017.csv"
sensor_data = pd.read_csv(os.path.join(path, sensor_file), delimiter=",")
sensor_data["time"] = pd.to_datetime(sensor_data["Data"], format = date_format) 
sensor_data = sensor_data.set_index(sensor_data["time"])
#sensor_data = sensor_data.replace({-9999: np.nan})
station_all_data_LOM = pd.DataFrame()
# combine var values with sensor id, lon, lat and aggregate to hourly values
for sensor_id in var_stations["IdSensore"]: 
    #print('sensor id:', sensor_id)
    sensor_df = sensor_data[sensor_data["IdSensore"]==sensor_id][['Valore']].sort_index()
    # for precipitation hourly sum
    if var_name_it == "Precipitazione":
        sensor_df_mean = sensor_df.resample("1H").sum(min_count=6)
    # for temperature hourly mean
    elif var_name_it == "Temperatura": 
        sensor_df_enough_timesteps = (sensor_df.resample("1H").count())
        sensor_df_enough_timesteps = sensor_df_enough_timesteps[sensor_df_enough_timesteps.Valore==6]
        sensor_df_mean_all = sensor_df.resample("1H").mean()
        sensor_df_mean = sensor_df_mean_all[sensor_df_mean_all.index.isin(sensor_df_enough_timesteps.index)].sort_index()
    else: 
        print('var_name_it '+str(var_name_it)+' unknown!')
    sensor_df_mean['IdSensore'] = [sensor_id]*len(sensor_df_mean)
    lon, lat = station_data[station_data["IdSensore"]==sensor_id][['lng','lat']].values[0]
    sensor_df_mean['lon'] = [lon]*len(sensor_df_mean)
    sensor_df_mean['lat'] = [lat]*len(sensor_df_mean)
    # combine all sensor to a big df 
    station_all_data_LOM = pd.concat([station_all_data_LOM, sensor_df_mean])

#remove duplicates completely! in our case: 2377 and 2385 
station_all_data_LOM.rename(columns = {"Valore":"TEMP_station"}, inplace = True)
#1. take out weird values 
station_all_data_LOM.loc[(station_all_data_LOM.TEMP_station  > 100 ), "TEMP_station" ] = np.nan
#station_all_data_LOM = station_all_data_LOM[(station_all_data_LOM['IdSensore'] != 2377) & (station_all_data_LOM['IdSensore'] != 2385) & (station_all_data_LOM['IdSensore'] != 9106)]
# 2. set time range 
station_all_data_LOM = station_all_data_LOM.sort_index()
station_all_data_LOM = station_all_data_LOM["2017-03-01 00:00:00":"2017-07-31 23:00:00"]
# 3. drop dplicates 
station_all_data_LOM.drop_duplicates()
# 4. set time 
station_all_data_LOM['time'] = station_all_data_LOM.index
# 
# rotate
stations_rot = []
rot_longitude = []
rot_latitude = []
for i, (lon, lat) in enumerate(zip(station_all_data_LOM.lon, station_all_data_LOM.lat)): 
    stations_rot.append(rotated_coord_transform( lon, lat, 
    np_lon, np_lat, direction="geo2rot") )
    rot_longitude.append(round(stations_rot[i][0], 7))
    rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_LOM["rlon"] = rot_longitude
station_all_data_LOM["rlat"] = rot_latitude
station_all_data_LOM['region'] = ['Lombardia']*len(station_all_data_LOM)
station_all_data_LOM = station_all_data_LOM.drop(columns = 'IdSensore')
station_all_data_LOM


# VENTEO
# read data from Veneto 
import glob

station_path_VEN  = '/work/ch0636/g300099/data/station/VENETO'
station_files_VEN = glob.glob(os.path.join(station_path_VEN, '*Temperatura*.txt'))
station_data_VEN  = pd.concat((pd.read_csv(f, sep = ';', encoding = "ISO-8859-1") for f in station_files_VEN), ignore_index=True)
station_data_VEN['time'] = pd.to_datetime(station_data_VEN['DATA'].astype(str)  + ' ' +station_data_VEN['ORA'].astype(str), format = "%d/%m/%Y %H:%M")
station_data_VEN  = station_data_VEN.rename(columns={'STAZIONE':'name','VALORE':'TEMP_station'}).drop(columns=(['DATA','ORA','SENSORE']))

# 
station_detail_file_VEN = "Meteological Stations ARPA VENETO  - Dott.ssa ASMUS.xlsx"
station_detail = pd.read_excel(os.path.join(station_path_VEN, station_detail_file_VEN ),sheet_name = 'Foglio2',skiprows = 11,decimal='.' )
station_detail = station_detail.rename(columns={'Unnamed: 0':'Code','Unnamed: 1':'name','dal':'start date','Unnamed: 3':'stop date',
                                              'Unnamed: 4':'type','Unnamed: 5':'quota metri','Unnamed: 6':'commune', 
                                              'Unnamed: 7':'Gauss_x','Unnamed: 8':'Gauss_y','Unnamed: 9':'geo_lat','Unnamed: 10':'geo_lon', 
                                              'Unnamed: 11':'lat_deg','Unnamed: 12':'lat_min','Unnamed: 13':'lat_sec','Unnamed: 14':'lon_deg',
                                              'Unnamed: 15':'lon_min','Unnamed: 16':'lon_sec','Unnamed: 17':'basin','Unnamed: 18':'sotto basin'
                                              })
station_detail = station_detail.drop(columns=(['Code','type', 'quota metri','commune','Gauss_x','Gauss_y','geo_lat','geo_lon','basin','sotto basin']))
station_detail['lat_sec'] = station_detail['lat_sec']/1000
station_detail['lon_sec'] = station_detail['lon_sec']/1000

degrees = station_detail['lat_deg']
minutes = station_detail['lat_min']
seconds = station_detail['lat_sec']
station_detail['lat']=(degrees + ((minutes/60) + seconds/(60*60)))
degrees = station_detail['lon_deg']
minutes = station_detail['lon_min']
seconds = station_detail['lon_sec']
station_detail['lon']=(degrees + ((minutes/60) + seconds/(60*60)))
station_detail = station_detail.drop(columns=(['lat_deg','lat_min','lat_sec','lon_deg','lon_min','lon_sec']))

# 
station_all_data_VEN = station_data_VEN.merge(station_detail, how='left', on=['name'])
station_all_data_VEN = station_all_data_VEN.set_index('time', drop=True)
station_all_data_VEN = station_all_data_VEN.sort_index()
station_all_data_VEN = station_all_data_VEN['2017-03-01 00:00:00':'2017-07-31 23:00:00']
station_all_data_VEN
# 
# rotate
stations_rot = []
rot_longitude = []
rot_latitude = []
for i, (lon, lat) in enumerate(zip(station_all_data_VEN.lon, station_all_data_VEN.lat)): 
   stations_rot.append(rotated_coord_transform( lon, lat, 
   np_lon, np_lat, direction="geo2rot") )
   rot_longitude.append(round(stations_rot[i][0], 7))
   rot_latitude.append(round(stations_rot[i][1], 7))
station_all_data_VEN["rlon"] = rot_longitude
station_all_data_VEN["rlat"] = rot_latitude
station_all_data_VEN['region'] = ['Veneto']*len(station_all_data_VEN)
station_all_data_VEN = station_all_data_VEN.drop(columns=(['name','start date','stop date']))
station_all_data_VEN['time'] = station_all_data_VEN.index
station_all_data_VEN


# merge ER, LOM, and VEN 
df_all_station = pd.concat([station_all_data_ER, station_all_data_LOM, station_all_data_VEN]) 
print('expected length', (len(station_all_data_ER) + len(station_all_data_LOM)+len(station_all_data_VEN)))
df_all_station_clean = df_all_station.drop_duplicates()
df_all_station_clean


df_all_station_clean_in = df_all_station_clean.where((df_all_station_clean.rlon>sorted(po_noirri_SGAR.rlon.values)[0])
                                                     &(df_all_station_clean.rlon<sorted(po_noirri_SGAR.rlat.values)[-1])
                                                     &(df_all_station_clean.rlat>sorted(po_noirri_SGAR.rlat.values)[0])
                                                     &(df_all_station_clean.rlat<sorted(po_noirri_SGAR.rlat.values)[-1]))
df_all_station_temp = df_all_station_clean_in

print('Stations unique:', len(df_all_station_clean_in.rlon.unique()))
print('Stations unique:', len(df_all_station_clean_in.rlat.unique()))

############################ Test station numbers ################################


test_temp = df_all_station_temp[(~df_all_station_temp.rlon.isin(precip_temp_station_rlon))&(~df_all_station_temp.rlat.isin(precip_temp_station_rlat))]
test1 = test_temp.dropna()
len(test1.rlon.unique())

test_precip = df_all_station_precip[(~df_all_station_precip.rlon.isin(precip_temp_station_rlon))&(~df_all_station_precip.rlat.isin(precip_temp_station_rlat))]
test2 = test_precip.dropna()
len(test2.rlon.unique())

test3 = df_all_station_precip.dropna()
len(test3.rlon.unique() )
#    --> here the precip & temp-values are  still together 

#lon_list = df_all_station_clean_in.rlon.unique()
#lat_list = df_all_station_clean_in.rlat.unique()

# find stations which measure precipitation and temperature 
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
precip_temp_station_rlon =  intersection(df_all_station_precip.rlon.unique(),df_all_station_temp.rlon.unique())
precip_temp_station_rlat =  intersection(df_all_station_precip.rlat.unique(),df_all_station_temp.rlat.unique())

################################# create plot ###################################
# topography 
fib_GAR_complete = dsnoirr_newtime_GAR.FIB.squeeze('time')
fib_GAR = dsnoirr_newtime_GAR_cut.FIB.squeeze('time')
fib_SGAR = dsnoirr_newtime_SGAR.FIB.squeeze('time')

x_SGAR = [fib_SGAR.rlon[0], fib_SGAR.rlon[-1], fib_SGAR.rlon[-1], fib_SGAR.rlon[0], fib_SGAR.rlon[0]]
y_SGAR = [fib_SGAR.rlat[0], fib_SGAR.rlat[0], fib_SGAR.rlat[-1], fib_SGAR.rlat[-1], fib_SGAR.rlat[0]]
x_po = [po_noirri_SGAR.rlon[0], po_noirri_SGAR.rlon[-1], po_noirri_SGAR.rlon[-1], po_noirri_SGAR.rlon[0], po_noirri_SGAR.rlon[0]]
y_po = [po_noirri_SGAR.rlat[0], po_noirri_SGAR.rlat[0], po_noirri_SGAR.rlat[-1], po_noirri_SGAR.rlat[-1], po_noirri_SGAR.rlat[0]]
# irrifrac
irrifrac_GAR_complete = dsnoirr_newtime_GAR.IRRIFRAC.squeeze('time')
irrifrac_GAR = dsnoirr_newtime_GAR_cut.IRRIFRAC.squeeze('time')
irrifrac_SGAR = dsnoirr_newtime_SGAR.IRRIFRAC.squeeze('time')

irrifrac_GAR_complete = irrifrac_GAR_complete.where(irrifrac_GAR_complete>0)
irrifrac_GAR = irrifrac_GAR.where(irrifrac_GAR>0)
irrifrac_SGAR = irrifrac_SGAR.where(irrifrac_SGAR>0)

params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
# plot
fig = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(6, 3)
plt.rcParams.update(params)

cax1 = fig.add_axes([0.125, 0.27, 0.265, 0.03])
cax2 = fig.add_axes([0.4, 0.27, 0.23, 0.03])

levels_fib = np.arange(0,3200,200)
ticks_fib = levels_fib[::5]
levels_irrifrac = np.arange(0,1.1,0.1)
ticks_irrifrac = np.round(levels_irrifrac[::2],1)

# topography
ax1 = fig.add_subplot(gs[0:6, 0], projection=rotated_pole)
ax1.set_aspect('equal')       
ax1.set_axis_off()
rotplot1 = plot_topography(fig,  fib_GAR_complete, ax1, cax1, 'm', 'elevation [m]', levels_fib, ticks_fib, 'horizontal') 
rotplot2 = plot_topography(fig,  fib_SGAR, ax1, cax1, 'm', 'elevation [m]', levels_fib, ticks_fib, 'horizontal') 
ax1.plot(x_SGAR, y_SGAR, color='blue', linewidth=1.5, transform=rotated_pole, zorder = 4 ) 
ax1.plot(x_po, y_po, color='red', linewidth=1.5, transform=rotated_pole, zorder = 4 )
ax1.text(0.035, 0.964, 'D11', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', bbox=dict(facecolor='white',alpha=0.8, edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.377, 0.682, 'D0275', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8,edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.498, 0.445, 'D0275-EVAL', transform=ax1.transAxes, fontsize=7.5, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=1), zorder=+6)
ax1.text(0.0, 1.063, '(a)', transform=ax1.transAxes, fontsize=16)
ax1.set_title('')

# irrifrac 
levels = np.arange(0,1.05,0.05)
ticks = (levels[::4]).round(1)
ax2 = fig.add_subplot(gs[0:6, 1], projection=rotated_pole)
ax2.set_aspect('equal')
ax2.set_axis_off()
rotplot3 = plot_rotvar_adjust_cbar(fig, irrifrac_SGAR, ax2, cax2, '-', 'irrigated fraction [-]', 'viridis_r', levels, ticks, 'neither', False,'horizontal')
ax2.plot(x_SGAR, y_SGAR, color='blue', linewidth=2, transform=rotated_pole, zorder = 4 )
ax2.plot(x_po, y_po, color='red', linewidth=1.5, transform=rotated_pole, zorder = 4 )
ax2.text(0.035, 0.964, 'D0275', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top', bbox=dict(facecolor='white',alpha=0.8, edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.29, 0.48, 'D0275-EVAL', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha = 0.8, edgecolor='black', pad=3.0), zorder=+6)
ax2.text(0.0, 1.035, '(b)', transform=ax2.transAxes, fontsize=16)
ax2.set_title('')

ax3 = fig.add_subplot(gs[0:5, 2], projection=rotated_pole)
plt.scatter(
marker = 'x',
    #x=df_all_station_precip.rlon.unique(),
    #y=df_all_station_precip.rlat.unique(),
    x = test2.rlon.unique(),
    y = test2.rlat.unique(),
    s=13,
    facecolors='cornflowerblue',
    alpha=0.7,
    transform=rotated_pole,
    edgecolor = None,
    label = 'precipitation',
)
plt.scatter(
    marker = 'x',
    #x=df_all_station_temp.rlon.unique(),
    #y=df_all_station_temp.rlat.unique(),
    x = test1.rlon.unique(),
    y = test1.rlat.unique(),
    s=15,
    facecolors='fuchsia',
    #alpha=0.5,
    transform=rotated_pole,
    edgecolor = None,
    label = 'temperature'
)
plt.scatter(
    marker = 'd',
    x=precip_temp_station_rlon,
    y=precip_temp_station_rlat,
    s=20,
    facecolors='orange',
    transform=rotated_pole,
    edgecolor = None,
    label = 'both',
    alpha = 0.5
)

ax3.add_feature(land, zorder=1)
ax3.add_feature(coastline, edgecolor="black", linewidth=0.6)
ax3.add_feature(borders, edgecolor="black", linewidth=0.6)
ax3.set_xmargin(0)
ax3.set_ymargin(0)
ax3.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
ax3.plot(x_po, y_po, color='red', linewidth=2, transform=rotated_pole, zorder = 4 )
ax3.legend(markerscale=2, bbox_to_anchor= (0.538, 0.0))
ax3.text(0.02, 0.12, 'D0275-EVAL', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha = 0.8, edgecolor='black', pad=3.0), zorder=+6)
ax3.text(0.0, 1.1, '(c)', transform=ax3.transAxes, fontsize=16)
ax3.set_title('')
#plt.savefig(str(dir_out)+'/all_stations_topo_irrifrac_new.png',dpi=300, bbox_inches='tight')




