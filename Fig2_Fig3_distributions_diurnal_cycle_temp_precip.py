#!/usr/bin/env python
# coding: utf-8

import os
import glob
import csv
import warnings

from datetime import datetime
from datetime import date

import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as stats

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from functions_correcting_time import correct_timedim
from functions_plotting import rotated_pole
from functions_reading_files import read_efiles, read_efiles_in_chunks, cut_same_area 
from functions_rotation import rotated_coord_transform, np_lon, np_lat
from functions_idw import idw_for_model

print('working directory',os.getcwd())
# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

####################### READ REMO DATA ############################################

# temperature and precipitation
# read temperature and precipitation
year = 2017
month = 0
#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"
#
varlist_SGAR = [ "TEMP2", "APRL"]
var_num_list_SGAR = [ "167","142" ]
#
varlist_GAR = [  "TEMP2", "APRL", "APRC"]
var_num_list_GAR = ["167","142", "143" ]


# adapt your paths where the data is stored.
# Please also check the functions in functions_reading_files.py. 
data_path_SGAR_noirri = "/noirri0275"
data_path_SGAR_irri = "/irri0275"
data_path_GAR_noirri = "/noirri11"
data_path_GAR_irri = "/irri11"

# SGAR 
for var_SGAR, var_num_SGAR in zip(varlist_SGAR, var_num_list_SGAR):
    print('reading ', var_SGAR)
    single_var_data_SGAR_irri = read_efiles_in_chunks(data_path_SGAR_irri, var_SGAR, var_num_SGAR, exp_number_irri_SGAR, year, month, 5, 10, None, None)
    single_var_data_SGAR_noirri = read_efiles_in_chunks(data_path_SGAR_noirri, var_SGAR, var_num_SGAR, exp_number_noirri_SGAR, year, month, 5, 10, None, None)
   
    if var_SGAR == varlist_SGAR[0]:
        ds_var_irri_SGAR = single_var_data_SGAR_irri
        ds_var_noirri_SGAR = single_var_data_SGAR_noirri
    else:
        ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, single_var_data_SGAR_irri])
        ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, single_var_data_SGAR_noirri])
# GAR
for var_GAR, var_num_GAR in zip(varlist_GAR, var_num_list_GAR):
    print('reading ', var_GAR)
    single_var_data_GAR_irri = read_efiles(data_path_GAR_irri, var_GAR, var_num_GAR, exp_number_irri_GAR, year, month)
    single_var_data_GAR_noirri = read_efiles(data_path_GAR_noirri, var_GAR, var_num_GAR, exp_number_noirri_GAR, year, month)
   
    if var_GAR == varlist_GAR[0]:
        ds_var_irri_GAR = single_var_data_GAR_irri
        ds_var_noirri_GAR = single_var_data_GAR_noirri
    else:
        ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, single_var_data_GAR_irri])
        ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, single_var_data_GAR_noirri])

#adding elevation 
elev_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/FIB/e'+str(exp_number_noirri_SGAR)+'e_c129_201706.nc'
elev_data_SGAR = xr.open_dataset(elev_file_SGAR)
elev_SGAR = elev_data_SGAR.FIB[0]

elev_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/FIB/e'+str(exp_number_noirri_GAR)+'e_c129_201706.nc'
elev_data_GAR = xr.open_dataset(elev_file_GAR)
elev_GAR = elev_data_GAR.FIB[0]


# adding irrifrac 
irrifrac_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_SGAR)+'e_c743_201706.nc'
irrifrac_data_SGAR = xr.open_dataset(irrifrac_file_SGAR)
irrifrac_SGAR = irrifrac_data_SGAR.IRRIFRAC[0]

irrifrac_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_GAR)+'e_c743_201706.nc'
irrifrac_data_GAR = xr.open_dataset(irrifrac_file_GAR)
irrifrac_GAR = irrifrac_data_GAR.IRRIFRAC[0]

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR, elev_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR, elev_SGAR])

ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR, elev_GAR ])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR, elev_GAR])

dsirr_newtime_SGAR = correct_timedim(ds_var_irri_SGAR)
dsnoirr_newtime_SGAR = correct_timedim(ds_var_noirri_SGAR)

dsirr_newtime_GAR = correct_timedim(ds_var_irri_GAR)
dsnoirr_newtime_GAR = correct_timedim(ds_var_noirri_GAR)

# hier schneiden wir GAR auf das Gebiet von SGAR zu. 
# cut out the same region 
dsnoirr_newtime_GAR_cut = cut_same_area(dsnoirr_newtime_GAR, dsnoirr_newtime_SGAR)
dsirr_newtime_GAR_cut = cut_same_area(dsirr_newtime_GAR, dsirr_newtime_SGAR)

 # select months
dsnoirr_newtime_GAR_cut = dsnoirr_newtime_GAR_cut.sel(time=dsnoirr_newtime_GAR_cut.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_GAR_cut   = dsirr_newtime_GAR_cut.sel(time=dsirr_newtime_GAR_cut.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_SGAR     = dsirr_newtime_SGAR.sel(time=dsirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsnoirr_newtime_SGAR   = dsnoirr_newtime_SGAR.sel(time=dsnoirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_GAR_cut


# cut out the Po valley 
po_irri_GAR    = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_GAR  = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))

po_noirri_SGAR = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR)
po_irri_SGAR   = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR)

 # select months
po_irri_GAR    = po_irri_GAR.sel(time=po_irri_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))
po_noirri_GAR  = po_noirri_GAR.sel(time=po_noirri_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))

po_irri_SGAR   = po_irri_SGAR.sel(time=po_irri_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
po_noirri_SGAR = po_noirri_SGAR.sel(time=po_noirri_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
po_irri_GAR

# we have to split/resample the 12 km simulations with nearest neighbor 
# split GAR to be able to use the filter 

dsnoirr_newtime_GAR_split = dsnoirr_newtime_GAR_cut.interp_like(
    dsnoirr_newtime_SGAR,
    method='nearest')
dsirr_newtime_GAR_split = dsirr_newtime_GAR_cut.interp_like(
    dsirr_newtime_SGAR,
    method='nearest')

############################select topography and irrifrac ###################
# values from static variables: irrifrac and FIB 
# for histogram plot
irrifrac_GAR       = dsnoirr_newtime_GAR_cut.IRRIFRAC.where(dsnoirr_newtime_GAR_cut.IRRIFRAC>0)
irrifrac_GAR_split = dsnoirr_newtime_GAR_split.IRRIFRAC.where(dsnoirr_newtime_GAR_split.IRRIFRAC>0)
irrifrac_SGAR      = dsnoirr_newtime_SGAR.IRRIFRAC.where(dsnoirr_newtime_SGAR.IRRIFRAC>0)
fib_GAR            = dsnoirr_newtime_GAR_cut.FIB
fib_GAR_split      = dsnoirr_newtime_GAR_split.FIB
fib_SGAR           = dsnoirr_newtime_SGAR.FIB

irrifrac_GAR_1d = irrifrac_GAR.values.reshape(len(irrifrac_GAR.rlon)*len(irrifrac_GAR.rlat))
irrifrac_SGAR_1d = irrifrac_SGAR.values.reshape(len(irrifrac_SGAR.rlon)*len(irrifrac_SGAR.rlat))
irrifrac_GAR_split_1d = irrifrac_GAR_split.values.reshape(len(irrifrac_GAR_split.rlon)*len(irrifrac_GAR_split.rlat))

fib_GAR_1d = fib_GAR.values.reshape(len(fib_GAR.rlon)*len(fib_GAR.rlat))
fib_SGAR_1d = fib_SGAR.values.reshape(len(fib_SGAR.rlon)*len(fib_SGAR.rlat))
fib_GAR_split_1d = fib_GAR_split.values.reshape(len(fib_GAR_split.rlon)*len(fib_GAR_split.rlat))


#############READ STATION DATA ##########################################
############# PRECIPITATION ##############################################

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
number_of_ts = 5142-4
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
station_all_data_LOM = station_all_data_LOM.sort_index()
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

station_path_VEN = '/work/ch0636/g300099/data/station/VENETO'
station_files_VEN = glob.glob(os.path.join(station_path_VEN, '*Precipitazione*.txt'))
station_data_VEN  = pd.concat((pd.read_csv(f, sep = ';', encoding = "ISO-8859-1") for f in station_files_VEN), ignore_index=True)
station_data_VEN['time'] = pd.to_datetime(station_data_VEN['DATA'].astype(str)  + ' ' +station_data_VEN['ORA'].astype(str), format = "%d/%m/%Y %H:%M")
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

##### COMBINE THE STATION DATA ########################
# merge ER, LOM, and VEN 
df_all_station = pd.concat([station_all_data_ER, station_all_data_LOM, station_all_data_VEN]) 
print('expected length', (len(station_all_data_ER) + len(station_all_data_LOM)+len(station_all_data_VEN)))
df_all_station_clean = df_all_station.drop_duplicates()
df_all_station_precip = df_all_station_clean

df_all_station_precip_new = df_all_station_precip.replace(99999.0, np.nan)
#df_all_station_precip_new.PRECIP_station.max()
plot_df = df_all_station_precip_new.dropna(axis = 0)
df_all_station_clean_new = df_all_station_precip_new

#### INTERPOLATE REMO DATA TO STATION DATA WITH IDW #############
# interpolation of REMO data to station lat, lon with idw 

po_irri_GAR['PRECIP'] = po_irri_GAR['APRL'] + po_irri_GAR['APRC']
po_noirri_GAR['PRECIP'] = po_noirri_GAR['APRL'] + po_noirri_GAR['APRC']

po_irri_SGAR['PRECIP'] = po_irri_SGAR['APRL'] 
po_noirri_SGAR['PRECIP'] = po_noirri_SGAR['APRL'] 

rlat_source_irri_GAR = po_irri_GAR.rlat.values 
rlon_source_irri_GAR = po_irri_GAR.rlon.values 
rlat_source_noirri_GAR = po_noirri_GAR.rlat.values 
rlon_source_noirri_GAR = po_noirri_GAR.rlon.values 

rlat_source_irri_SGAR = po_irri_SGAR.rlat.values 
rlon_source_irri_SGAR = po_irri_SGAR.rlon.values 
rlat_source_noirri_SGAR = po_noirri_SGAR.rlat.values 
rlon_source_noirri_SGAR = po_noirri_SGAR.rlon.values 

rlat_target = df_all_station_clean_new.rlat.unique()
rlon_target = df_all_station_clean_new.rlon.unique()

df_all_timestep = pd.DataFrame()
df_timestep_idw = pd.DataFrame()

for t in range(len(po_irri_GAR.time)):
    if t in [100,500,1000,1500,2000,2500,3000,3500]:
        print('timestep',t, 'out of ', len(po_irri_GAR.time))
    data_irri_GAR = po_irri_GAR['PRECIP'][t].values
    data_noirri_GAR = po_noirri_GAR['PRECIP'][t].values
    data_irri_SGAR = po_irri_SGAR['PRECIP'][t].values
    data_noirri_SGAR = po_noirri_SGAR['PRECIP'][t].values

    df_timestep_idw['time'] = ([po_irri_GAR.time.values[t]]*len(rlon_target))
    df_timestep_idw["rlon"] = rlon_target
    df_timestep_idw["rlat"] = rlat_target
    df_timestep_idw['PRECIP_model_irri_GAR']    = idw_for_model(rlat_source_irri_GAR, rlon_source_irri_GAR, rlat_target, rlon_target, data_irri_GAR, 'PRECIP_model_irri_GAR')
    df_timestep_idw['PRECIP_model_noirri_GAR']  = idw_for_model(rlat_source_noirri_GAR, rlon_source_noirri_GAR, rlat_target, rlon_target, data_noirri_GAR, 'PRECIP_model_noirri_GAR')
    df_timestep_idw['PRECIP_model_irri_SGAR']   = idw_for_model(rlat_source_irri_SGAR, rlon_source_irri_SGAR, rlat_target, rlon_target, data_irri_SGAR, 'PRECIP_model_irri_SGAR')
    df_timestep_idw['PRECIP_model_noirri_SGAR'] = idw_for_model(rlat_source_noirri_SGAR, rlon_source_noirri_SGAR, rlat_target, rlon_target, data_noirri_SGAR, 'PRECIP_model_noirri_SGAR')

    df_all_timestep = pd.concat([df_all_timestep, df_timestep_idw])

precip_df_combined = df_all_timestep.merge(df_all_station_clean_new, how='left', on=['time','rlon','rlat'])
precip_df_combined_cleaned = precip_df_combined.dropna()
precip_df_combined_cleaned = precip_df_combined_cleaned.drop_duplicates()
precip_df_combined_cleaned = precip_df_combined_cleaned.replace(99999.0, np.nan)
precip_df_combined_cleaned = precip_df_combined_cleaned.dropna(axis = 0)
precip_df_combined_cleaned

# distribution 
# for all months and all stations
precip_plot_irri_GAR     = precip_df_combined_cleaned[(precip_df_combined_cleaned['PRECIP_model_irri_GAR'] > 0)]['PRECIP_model_irri_GAR']
precip_plot_noirri_GAR   = precip_df_combined_cleaned[(precip_df_combined_cleaned['PRECIP_model_noirri_GAR'] > 0)]['PRECIP_model_noirri_GAR']
precip_plot_irri_SGAR    = precip_df_combined_cleaned[(precip_df_combined_cleaned['PRECIP_model_irri_SGAR'] > 0)]['PRECIP_model_irri_SGAR']
precip_plot_noirri_SGAR  = precip_df_combined_cleaned[(precip_df_combined_cleaned['PRECIP_model_noirri_SGAR'] > 0)]['PRECIP_model_noirri_SGAR']
precip_plot_station      = precip_df_combined_cleaned[(precip_df_combined_cleaned['PRECIP_station'] > 0)]['PRECIP_station']

############################  TEMPERATURE       #################################################
############################  READ STATION DATA #################################################

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
station_all_data_LOM = station_all_data_LOM["2017-03-01 00:00:00":"2017-08-31 23:00:00"]
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

# VENETEO
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
station_all_data_VEN = station_all_data_VEN['2017-03-01 00:00:00':'2017-08-31 23:00:00']
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

##################### COMBINE STATION DATA #################################

# merge ER, LOM, and VEN 
df_all_station = pd.concat([station_all_data_ER, station_all_data_LOM, station_all_data_VEN]) 
print('expected length', (len(station_all_data_ER) + len(station_all_data_LOM)+len(station_all_data_VEN)))
df_all_station_clean = df_all_station.drop_duplicates()
df_all_station_clean
df_all_station_temp = df_all_station_clean


##################### INTERPOLATE REMO DATA #################################

# interpolation of REMO data to station lat, lon we use idw 
rlat_source_irri_GAR = po_irri_GAR.rlat.values 
rlon_source_irri_GAR = po_irri_GAR.rlon.values 
rlat_source_noirri_GAR = po_noirri_GAR.rlat.values 
rlon_source_noirri_GAR = po_noirri_GAR.rlon.values 

rlat_source_irri_SGAR = po_irri_SGAR.rlat.values 
rlon_source_irri_SGAR = po_irri_SGAR.rlon.values 
rlat_source_noirri_SGAR = po_noirri_SGAR.rlat.values 
rlon_source_noirri_SGAR = po_noirri_SGAR.rlon.values 

rlat_target = df_all_station_clean.rlat.unique()
rlon_target = df_all_station_clean.rlon.unique()

df_all_timestep = pd.DataFrame()
df_timestep_idw = pd.DataFrame()

for t in range(len(po_irri_GAR.time)):
    if t in [100,500,1000,1500,2000,2500,3000,3500]:
        print('timestep',t, 'out of ', len(po_irri_GAR.time))
    data_irri_GAR = po_irri_GAR['TEMP2'][t].values
    data_noirri_GAR = po_noirri_GAR['TEMP2'][t].values
    data_irri_SGAR = po_irri_SGAR['TEMP2'][t].values
    data_noirri_SGAR = po_noirri_SGAR['TEMP2'][t].values

    df_timestep_idw['time'] = ([po_irri_GAR.time.values[t]]*len(rlon_target))
    df_timestep_idw["rlon"] = rlon_target
    df_timestep_idw["rlat"] = rlat_target
    df_timestep_idw['TEMP_model_irri_GAR']    = idw_for_model(rlat_source_irri_GAR, rlon_source_irri_GAR, rlat_target, rlon_target, data_irri_GAR, 'TEMP_model_irri_GAR')
    df_timestep_idw['TEMP_model_irri_GAR']    = df_timestep_idw['TEMP_model_irri_GAR']-273.15
    df_timestep_idw['TEMP_model_noirri_GAR']  = idw_for_model(rlat_source_noirri_GAR, rlon_source_noirri_GAR, rlat_target, rlon_target, data_noirri_GAR, 'TEMP_model_noirri_GAR')
    df_timestep_idw['TEMP_model_noirri_GAR']  = df_timestep_idw['TEMP_model_noirri_GAR']-273.15
    df_timestep_idw['TEMP_model_irri_SGAR']   = idw_for_model(rlat_source_irri_SGAR, rlon_source_irri_SGAR, rlat_target, rlon_target, data_irri_SGAR, 'TEMP_model_irri_SGAR')
    df_timestep_idw['TEMP_model_irri_SGAR']   = df_timestep_idw['TEMP_model_irri_SGAR']-273.15
    df_timestep_idw['TEMP_model_noirri_SGAR'] = idw_for_model(rlat_source_noirri_SGAR, rlon_source_noirri_SGAR, rlat_target, rlon_target, data_noirri_SGAR, 'TEMP_model_noirri_SGAR')
    df_timestep_idw['TEMP_model_noirri_SGAR'] = df_timestep_idw['TEMP_model_noirri_SGAR']-273.15

    df_all_timestep = pd.concat([df_all_timestep, df_timestep_idw])

# clean 1
temp_df_combined = df_all_timestep.merge(df_all_station_clean, how='left', on=['time','rlon','rlat'])
temp_df_combined_cleaned = temp_df_combined.dropna()
temp_df_combined_cleaned = temp_df_combined_cleaned.drop_duplicates()
temp_df_combined_cleaned = temp_df_combined_cleaned.replace(99999.0, np.nan)
temp_df_combined_cleaned

temp_plot_irri_GAR     = temp_df_combined_cleaned['TEMP_model_irri_GAR']
temp_plot_noirri_GAR   = temp_df_combined_cleaned['TEMP_model_noirri_GAR']
temp_plot_irri_SGAR    = temp_df_combined_cleaned['TEMP_model_irri_SGAR']
temp_plot_noirri_SGAR  = temp_df_combined_cleaned['TEMP_model_noirri_SGAR']
temp_plot_station      = temp_df_combined_cleaned['TEMP_station']

# clean 2 for unphysical values 
temp_plot_station_new = temp_plot_station[temp_plot_station>-273.15]
temp_df_combined_cleaned_new = temp_df_combined_cleaned[temp_df_combined_cleaned['TEMP_station']>-273.15]
temp_df_combined_cleaned_new

################################ PLOT DISTRIBUTIONS #########################################
# functions

def add_patches_to_hist(ax, colors, hatches, hatchcolors, legend, legend_entry, legend_title):
    colors = colors
    hatches = hatches
    hatchcolors = hatchcolors

    iter_colors = np.repeat(colors,len(bins)-1)
    iter_hatches = np.tile(np.repeat(hatches, len(bins)-1),2)
    iter_hatchcolors = np.repeat(hatchcolors,len(bins)-1)

    for patch,color, hatch, hatchcolor in zip(ax.patches, iter_colors, iter_hatches, iter_hatchcolors):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)
        patch.set_edgecolor(hatchcolor)
    if legend == True:
        # Add legends:
        res = legend_entry
    
        res_legend = ax.legend(
            [Patch(hatch=hatch, facecolor='white', edgecolor = 'black') for hatch in hatches],
            res, 
            bbox_to_anchor=(1.32, 1.02),
            loc = 'upper right',
            title=legend_title, labelspacing=.65)


def plot_density_histogram(fig, ax2, data1_irri, data1_noirri, data2_irri, data2_noirri, data_obs, bins, sim_legend, res_legend, params):  
    ## precipitation    

    plt.rcParams.update(params)

    hist_plot = ax2.hist([\
              np.clip(data1_irri, bins[0], bins[-1]), \
              np.clip(data1_noirri, bins[0], bins[-1]),\
              np.clip(data2_irri, bins[0], bins[-1]), \
              np.clip(data2_noirri, bins[0], bins[-1]),\
              np.clip(data_obs, bins[0], bins[-1])],\
              bins=bins, histtype='bar', density = True)
  


    colors = ['cornflowerblue', 'darkorange', 'white', 'white','dimgrey']
    hatches = [None,None, '///','///',None]
    hatchcolors = [None, None, 'cornflowerblue', 'darkorange',None]

    iter_colors = np.repeat(colors,len(bins)-1)
    iter_hatches = np.tile(np.repeat(hatches, len(bins)-1),2)
    iter_hatchcolors = np.repeat(hatchcolors,len(bins)-1)

    for patch,color, hatch, hatchcolor in zip(ax2.patches, iter_colors, iter_hatches, iter_hatchcolors):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)
        patch.set_edgecolor(hatchcolor)
    if sim_legend == True: 
        # Add legends:
        sim = ['irrigated', 'not irrigated','observation']
        res = ['0.11°', '0.0275°']
        color_legend = ['cornflowerblue', 'darkorange','dimgrey']
        sim_legend = ax2.legend(
            [Line2D([0], [0], color=color, lw=4) for color in color_legend],
            sim, title='simulation', 
            loc = 'upper right', 
            bbox_to_anchor=(1.18, 3.4))
    if res_legend == True: 
        res_legend = ax2.legend(
            [Patch(hatch=hatch, facecolor='white', edgecolor = 'black') for hatch in hatches[1::2]],
            res, bbox_to_anchor=(1., 1.2), 
            title='resolution', labelspacing=.65)

        # for size of patch 
        for patch in sim_legend.get_patches():
            patch.set_height(10)
            patch.set_y(-3)

        ax2.add_artist(sim_legend)
    else: 
        pass
    plt.yscale("log")
    
    # adjust plot 
    plt.xticks(bins[::2])
    plt.ylabel('density')
    return hist_plot

# plot 
gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(13,12))
params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)

# topography
ax1 = fig.add_subplot(gs[0, 0])
bins = np.arange(0,3200,200)
# for normalization
weights=[np.ones_like(fib_GAR_1d)*100 / len(fib_GAR_1d), np.ones_like(fib_SGAR_1d)*100 / len(fib_SGAR_1d)]
# topography has to be clipped
fib_GAR_1d_clip = np.clip(fib_GAR_1d, bins[0], bins[-1])
fib_SGAR_1d_clip = np.clip(fib_SGAR_1d, bins[0], bins[-1])

ax1.hist([fib_GAR_1d_clip, fib_SGAR_1d_clip], bins, histtype='bar', density = True)#, weights=weights)
ax1.set_xlabel('elevation [m]')
ax1.set_ylabel('density')
add_patches_to_hist(ax1, colors, hatches, hatchcolors, False, legend_entry, legend_title)
bin_labels = ['0', '200', '400', '600', '800', '1000', '1200', '1400', '1600', '1800', '2000',
       '2200', '2400', '2600', '2800', '≥3000']
plt.xticks(bins[::3], bin_labels[::3])
plt.yscale('log')
ax1.text(0.0, 1.035, '(a)', transform=ax1.transAxes, fontsize=14)

colors = [ 'darkorange', 'white']
hatches = [None,'///']
hatchcolors = [None, 'darkorange']
legend_entry = ['0.11°', '0.0275°'] 
legend_title = 'resolution'

# irrigated fraction
ax2 = fig.add_subplot(gs[0, 1])
bins = np.arange(0,1.05,0.05)
# for normalization
weights=[np.ones_like(irrifrac_GAR_1d)*100 / len(irrifrac_GAR_1d), np.ones_like(irrifrac_SGAR_1d)*100 / len(irrifrac_SGAR_1d)]
ax2.hist([irrifrac_GAR_1d, irrifrac_SGAR_1d], bins, histtype='bar', density = True)# , weights=weights)
ax2.set_xlabel('irrigated fraction [-]')
ax2.set_ylabel('density [-]')
plt.yscale('log')
add_patches_to_hist(ax2, colors, hatches, hatchcolors, True, legend_entry, legend_title)
ax2.text(0.0, 1.035, '(b)', transform=ax2.transAxes, fontsize=14)

# temperature distribution
ax3 = fig.add_subplot(gs[1, 0:2])
bins = np.arange(-5,50,5)
p = plot_density_histogram(fig, ax3, temp_plot_irri_GAR, temp_plot_noirri_GAR, temp_plot_irri_SGAR, 
                           temp_plot_noirri_SGAR, temp_plot_station_new, bins, False, False, params )
plt.xlabel('2m temperature [°C]')
labels = [item.get_text() for item in ax3.get_xticklabels()]
#labels[-1] = '≥ 45'
labels[0] = ' ≤ -5'
ax3.set_xticklabels(labels)
ax3.text(0.0, 1.035, '(c)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(gs[2, 0:2])
steps = 2.5
max_val = 42.5
max_tick = max_val - steps
bins = np.arange(0,max_val, steps)
p = plot_density_histogram(fig, ax4, precip_plot_irri_GAR, precip_plot_noirri_GAR, 
                           precip_plot_irri_SGAR, precip_plot_noirri_SGAR, 
                           precip_plot_station, bins, True, False, params )
plt.xlabel('precipitation [mm hour$^{-1}$]')
labels = [item.get_text() for item in ax4.get_xticklabels()]
labels[-1] = '≥ '+str(int(max_tick))
ax4.set_xticklabels(labels)
ax4.set_ylim(10**(-6))
ax4.text(0.0, 1.035, '(d)', transform=ax4.transAxes, fontsize=14)

plt.subplots_adjust(hspace = 0.4)
#plt.savefig(str(dir_out)+'/density_distribution_large_new.png',dpi=300, bbox_inches='tight')


######################## SIGNIFICANCE TESTS ############################### 

# irrifrac and topography

# we have to use the splited data to be able to do a paired test 
var_GAR_list = [irrifrac_GAR_split_1d, fib_GAR_split_1d]
var_SGAR_list = [irrifrac_SGAR_1d, fib_SGAR_1d]
var_name_list = ['IRRIFRAC', 'TOPO']

for (var_GAR, var_SGAR, var_name) in zip(var_GAR_list, var_SGAR_list, var_name_list): 
    print(var_name)
    data_GAR = var_GAR
    data_SGAR = var_SGAR
    
    data_GAR_cleaned = data_GAR[~np.isnan(data_GAR) & ~np.isnan(data_SGAR)]
    data_SGAR_cleaned = data_SGAR[~np.isnan(data_GAR) & ~np.isnan(data_SGAR)]
    
    shapiro_GAR = stats.shapiro(data_GAR_cleaned)
    shapiro_SGAR = stats.shapiro(data_SGAR_cleaned)
    
    print(f"Shapiro test (GAR): p-value = {shapiro_GAR.pvalue:.4f}")
    print(f"Shapiro test (SGAR): p-value = {shapiro_SGAR.pvalue:.4f}")
    
    # Step 2: Choose the test
    if shapiro_GAR.pvalue > 0.05 and shapiro_SGAR.pvalue > 0.05:
        # If normality holds, perform t-test
        t_stat, p_value = stats.ttest_rel(data_GAR_cleaned, data_SGAR_cleaned)  # Use ttest
        print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        p_value = p_value
    else:
       # If data is NOT normal, use Wilcoxon Signed-Rank Test
        w_stat, w_p = stats.wilcoxon(data_GAR_cleaned,data_SGAR_cleaned, alternative='two-sided')
        print(f"Wilcoxon Signed-Rank Test: statistic = {w_stat:.4f}, p-value = {w_p:.4f}")
        p_value =w_p
    # Step 3: Interpret significance
    alpha = 0.05
    if p_value < alpha:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is NOT statistically significant (p ≥ 0.05).")

# TEMP and PRECIP: irri vs. noirri

var_irri_list = [temp_plot_irri_GAR, temp_plot_irri_SGAR, precip_plot_irri_GAR, precip_plot_irri_SGAR]
var_noirri_list = [temp_plot_noirri_GAR, temp_plot_noirri_SGAR, precip_plot_noirri_GAR, precip_plot_noirri_SGAR]
var_name_list = ['Temp GAR', 'Temp SGAR', 'Precip GAR', 'Precip SGAR']

print('Test irri vs noirri at both resolutions')
for (var_irri, var_noirri, var_name) in zip(var_irri_list, var_noirri_list, var_name_list): 
    print(var_name)
    # Assuming var_irri and var_noirri are your paired data (with irrigation and without irrigation)
    # Clean the data (remove NaNs)
    data_irri_cleaned = var_irri[~np.isnan(var_irri) & ~np.isnan(var_noirri)]
    data_noirri_cleaned = var_noirri[~np.isnan(var_irri) & ~np.isnan(var_noirri)]
    
    # Shapiro-Wilk test for normality
    shapiro_irri = stats.shapiro(data_irri_cleaned)
    shapiro_noirri = stats.shapiro(data_noirri_cleaned)
    
    print(f"Shapiro test (irri): p-value = {shapiro_irri.pvalue:.4f}")
    print(f"Shapiro test (noirri): p-value = {shapiro_noirri.pvalue:.4f}")
    
    # Step 2: Choose the test based on normality
    if shapiro_irri.pvalue > 0.05 and shapiro_noirri.pvalue > 0.05:
        # If both distributions are normal, perform a paired t-test (dependent samples)
        t_stat, p_value = stats.ttest_rel(data_irri_cleaned, data_noirri_cleaned)
        print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    else:
        # If either of the distributions is not normal, use the Wilcoxon Signed-Rank Test
        w_stat, w_p = stats.wilcoxon(data_irri_cleaned, data_noirri_cleaned, alternative='two-sided')
        print(f"Wilcoxon Signed-Rank Test: statistic = {w_stat:.4f}, p-value = {w_p:.4f}")
        p_value = w_p
    
    # Step 3: Interpret significance
    alpha = 0.05
    if p_value < alpha:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is NOT statistically significant (p ≥ 0.05).")
    
    print('*********************************************************************')

# TEMP and PRECIP: GAR vs. SGAR

var_GAR_list = [temp_plot_irri_GAR, temp_plot_noirri_GAR , precip_plot_irri_GAR, precip_plot_noirri_GAR]
var_SGAR_list = [temp_plot_irri_SGAR, temp_plot_noirri_SGAR, precip_plot_irri_SGAR, precip_plot_noirri_SGAR]
var_name_list = ['Temp irri', 'Temp noirri', 'Precip irri', 'Precip noirri']

print('Test GAR vs SGAR at both resolutions for TEMP and PRECIP')
for (var_GAR, var_SGAR, var_name) in zip(var_GAR_list, var_SGAR_list, var_name_list): 
    print(var_name)
    
    data_GAR_cleaned = var_GAR[~np.isnan(var_GAR) & ~np.isnan(var_SGAR)]
    data_SGAR_cleaned = var_SGAR[~np.isnan(var_GAR) & ~np.isnan(var_SGAR)]

    # Shapiro-Wilk test for normality
    shapiro_GAR = stats.shapiro(data_GAR_cleaned)
    shapiro_SGAR = stats.shapiro(data_SGAR_cleaned)
    
    print(f"Shapiro test (irri): p-value = {shapiro_GAR.pvalue:.4f}")
    print(f"Shapiro test (noirri): p-value = {shapiro_SGAR.pvalue:.4f}")

    # Step 2: Choose the test based on normality
    if shapiro_GAR.pvalue > 0.05 and shapiro_SGAR.pvalue > 0.05:
        # If both distributions are normal, perform an independent t-test
        t_stat, p_value = stats.ttest_rel(data_GAR_cleaned, data_SGAR_cleaned)
        print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    else:
        # If either of the distributions is not normal, use the Wilcoxon Signed-Rank test
        w_stat, w_p = stats.wilcoxon(data_GAR_cleaned, data_SGAR_cleaned, alternative='two-sided')
        print(f"Wilcoxon Signed-Rank Test: statistic = {mw_stat:.4f}, p-value = {mw_p:.4f}")
        p_value = w_p
    
    alpha = 0.05
    if p_value < alpha:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is NOT statistically significant (p ≥ 0.05).")
    

############# PLOT DIURNAL CYCLE ####################
        
# diurnal cycle 
temp_df_grouped_by_hour = temp_df_combined_cleaned_new[['TEMP_model_irri_GAR','TEMP_model_noirri_GAR','TEMP_model_irri_SGAR','TEMP_model_noirri_SGAR','TEMP_station']]\
    .groupby([temp_df_combined_cleaned_new.time.dt.hour]).mean()
# diurnal cycle 
precip_df_grouped_by_hour = precip_df_combined_cleaned[['PRECIP_model_irri_GAR','PRECIP_model_noirri_GAR','PRECIP_model_irri_SGAR','PRECIP_model_noirri_SGAR','PRECIP_station']]\
    .groupby([precip_df_combined_cleaned.time.dt.hour]).mean()


#plot
precip_color_dict = {'PRECIP_model_irri_GAR': 'cornflowerblue', 'PRECIP_model_noirri_GAR': 'darkorange', 'PRECIP_model_irri_SGAR': 'cornflowerblue',\
              'PRECIP_model_noirri_SGAR': 'darkorange', 'PRECIP_station': 'black'}

temp_color_dict = {'TEMP_model_irri_GAR': 'cornflowerblue', 'TEMP_model_noirri_GAR': 'darkorange', 'TEMP_model_irri_SGAR': 'cornflowerblue',\
              'TEMP_model_noirri_SGAR': 'darkorange', 'TEMP_station': 'black'}

linestyles = ['-','-','--','--','-']
legend_names =['0.11°, irrigated', '0.11°, not irrigated', '0.0275°, irrigated', '0.0275°, not irrigated', 'observation']

fig = plt.figure( figsize = (13,4))
params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)
# temperature
ax1 = fig.add_subplot(1,2,1)
temp_df_grouped_by_hour.plot(ax = ax1, color = temp_color_dict, style = linestyles, legend = False)
ax1.set_ylabel('2m temperature [°C]')
ax1.text(0.0, 1.035, '(a)', transform=ax1.transAxes, fontsize=14)
#precipitation
ax2 = fig.add_subplot(122)
precip_df_grouped_by_hour.plot(ax = ax2, color = precip_color_dict, style = linestyles)
ax2.set_ylabel('precipitation [mm hour$^{-1}$]')
plt.legend(legend_names)
ax2.text(0.0, 1.035, '(b)', transform=ax2.transAxes, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.26)
#plt.savefig(str(dir_out)+'/diurnal_cycle_new.png',dpi=300, bbox_inches='tight')

