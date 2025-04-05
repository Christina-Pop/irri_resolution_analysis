#!/usr/bin/env python
# coding: utf-8

import nzthermo as nzt
import metpy.calc as mpcalc
import metpy.calc
from metpy.units import units
from metpy.calc import cape_cin, lcl, lfc
from metpy.calc.thermo import dewpoint_from_specific_humidity, parcel_profile 

import os 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from functions_correcting_time import correct_timedim
from functions_plotting import plot_rotvar, rotated_pole
from functions_reading_files import read_efiles, read_efiles_in_chunks

# create plotting directory 
dir_working = os.getcwd()
# creates dir in parent directory
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

############# READ PRESSURE INTERPOLATED FILES #################################

pfiles_noirri_SGAR = '/noirri0275/pfiles/*.nc'
pfiles_irri_SGAR = '/irri0275/pfiles/*.nc'

pfiles_noirri_GAR = '/noirri11/pfiles/*.nc'
pfiles_irri_GAR = '/irri/pfiles/*.nc'

po_noirri_GAR = xr.open_mfdataset(pfiles_noirri_GAR)
po_irri_GAR = xr.open_mfdataset(pfiles_irri_GAR)

po_noirri_SGAR = xr.open_mfdataset(pfiles_noirri_SGAR)
po_irri_SGAR = xr.open_mfdataset(pfiles_irri_SGAR)
print('****Before split***')
print(po_irri_GAR.sizes)
print(po_noirri_GAR.sizes)
print(po_irri_SGAR.sizes)
print(po_noirri_SGAR.sizes)

po_noirri_GAR_split = po_noirri_GAR.interp_like(
    po_noirri_SGAR,
    method='nearest')
po_irri_GAR_split = po_irri_GAR.interp_like(
    po_irri_SGAR,
    method='nearest')

# adjust dimension order. This is very important for using nzt later
po_noirri_GAR_splitt = po_noirri_GAR_split.assign(
    {var: po_noirri_GAR_split[var].transpose( "time", "plev", "rlat", "rlon") for var in po_noirri_GAR_split.data_vars}
)

po_irri_GAR_splitt = po_irri_GAR_split.assign(
    {var: po_irri_GAR_split[var].transpose( "time", "plev", "rlat", "rlon", ) for var in po_irri_GAR_split.data_vars}
)

print('****After split***')
print(po_irri_GAR_splitt.sizes)
print(po_noirri_GAR_splitt.sizes)
print(po_irri_SGAR.sizes)
print(po_noirri_SGAR.sizes)



# adapt where you stored the data
# include PBL 
file_hor_GAR_irri = '/irri11/var_series/GHPBL/e067027e_c271_201706.nc'
file_hor_GAR_noirri = '/noirri11/var_series/GHPBL/e067026e_c271_201706.nc'
file_hor_SGAR_irri = '/irri0275/var_series/GHPBL/e067109e_c271_201706.nc'
file_hor_SGAR_noirri = '/noirri0275/var_series/GHPBL/e067108e_c271_201706.nc'

ds_var_irri_GAR = xr.open_mfdataset(file_hor_GAR_irri)
ds_var_noirri_GAR = xr.open_mfdataset(file_hor_GAR_noirri)
ds_var_irri_SGAR = xr.open_mfdataset(file_hor_SGAR_irri)
ds_var_noirri_SGAR = xr.open_mfdataset(file_hor_SGAR_noirri)

# adding irrifrac 
irrifrac_file_SGAR = '/work/ch0636/g300099/SIMULATIONS/SGAR0275/remo_results/067108/2017/hourly/var_series/IRRIFRAC/e067108e_c743_201706.nc'
irrifrac_data_SGAR = xr.open_dataset(irrifrac_file_SGAR)
irrifrac_SGAR = irrifrac_data_SGAR.IRRIFRAC[0]

irrifrac_file_GAR ='/work/ch0636/g300099/SIMULATIONS/GAR11/remo_results/067026/2017/hourly/var_series/IRRIFRAC/e067026e_c743_201706.nc'
irrifrac_data_GAR = xr.open_dataset(irrifrac_file_GAR)
irrifrac_GAR = irrifrac_data_GAR.IRRIFRAC[0]

topo_file_SGAR = '/work/ch0636/g300099/SIMULATIONS/SGAR0275/remo_results/067108/2017/hourly/var_series/FIB/e067108e_c129_201706.nc'
topo_data_SGAR = xr.open_dataset(topo_file_SGAR)
topo_SGAR = topo_data_SGAR.FIB[0]

topo_file_GAR ='/work/ch0636/g300099/SIMULATIONS/GAR11/remo_results/067026/2017/hourly/var_series/FIB/e067026e_c129_201706.nc'
topo_data_GAR = xr.open_dataset(topo_file_GAR)
topo_GAR = topo_data_GAR.FIB[0]

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR, topo_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR, topo_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR, topo_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR, topo_GAR])

dsirr_newtime_SGAR = correct_timedim(ds_var_irri_SGAR)
dsnoirr_newtime_SGAR = correct_timedim(ds_var_noirri_SGAR)
dsirr_newtime_GAR = correct_timedim(ds_var_irri_GAR)
dsnoirr_newtime_GAR = correct_timedim(ds_var_noirri_GAR)

# cut out the same region 
def cut_same_area(source_area, target_area):
    min_rlon_target=target_area.rlon[0].values 
    max_rlon_target=target_area.rlon[-1].values

    min_rlat_target=target_area.rlat[0].values 
    max_rlat_target=target_area.rlat[-1].values
    
    cutted_ds = source_area.sel(rlon=slice(min_rlon_target,max_rlon_target), \
                                rlat=slice(min_rlat_target,max_rlat_target))
    return cutted_ds

# cut out the Po valley 
po_irri_GAR_hor = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_GAR_hor = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_SGAR_hor = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR_hor)
po_irri_SGAR_hor = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR_hor)

 # select months
po_irri_GAR_hor    = po_irri_GAR_hor.sel(time=po_irri_GAR_hor.time.dt.month.isin([6]))
po_noirri_GAR_hor  = po_noirri_GAR_hor.sel(time=po_noirri_GAR_hor.time.dt.month.isin([6]))
po_irri_SGAR_hor  = po_irri_SGAR_hor.sel(time=po_irri_SGAR_hor.time.dt.month.isin([6]))
po_noirri_SGAR_hor   = po_noirri_SGAR_hor.sel(time=po_noirri_SGAR_hor.time.dt.month.isin([6]))

# split 0.11 in 0.0275 

po_noirri_GAR_hor_split = po_noirri_GAR_hor.interp_like(
    po_noirri_SGAR,
    method='nearest')
po_irri_GAR_hor_split = po_irri_GAR_hor.interp_like(
    po_irri_SGAR,
    method='nearest')

# create irrimask for PDF plots 
irrimask_SGAR = xr.where(po_irri_SGAR_hor.IRRIFRAC>0.7, 1, np.nan)


#################################### 1. CAPE and CIN ############################################
# 
# * was calculated with nzthermo
# * installed from https://github.com/leaver2000/nzthermo?tab=readme-ov-file#getting-started
# * example notebook https://github.com/leaver2000/nzthermo-notebooks/blob/master/notebook.ipynb
# * nzthermo is based on metpy, I think, a comparison can also be found here https://github.com/Unidata/MetPy/issues/3481 

def calculate_cape_cin(array, start_ts, end_ts):
    # Step1: get all the variables
    pressure = array.coords["plev"].to_numpy().astype(np.float32)  # (Pa) 
    temperature = (
        array["T"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    ) # K
    specific_humidity = (
        array["QD"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    )  # kg/kg

    # Step2: flatten the 4d data to 2d     # - weatherbench's levels are in reverse order
    # - non vertical dimensions are flattened like (T, Z, Y, X) -> (T*Y*X, Z) || (N, Z)
    P = pressure[::-1]
    Z = len(P)
    # therefore (N, Z) where N == T*Y*X
    T = np.moveaxis(temperature[:, ::-1, :, :], 1, -1).reshape(-1, Z)  # (N, Z)
    print(f"{temperature.shape} -> {T.shape} || (T, Z, Y, X) -> (N, Z)")
    Td = nzt.dewpoint_from_specific_humidity(
        P[np.newaxis, :],
        np.moveaxis(specific_humidity[:, ::-1, :, :], 1, -1).reshape(-1, Z),
    )  # (K) (N, Z)

    # Step3: calculate cape and cin
    T0 = T[:, 0]
    Td0 = Td[:, 0]
    TIME, LAT, LON = (temperature.shape[0],) + temperature.shape[2:]
    prof = nzt.parcel_profile(P, T0, Td0)  # (T, Y, X, Z)
    CAPE, CIN = nzt.cape_cin(P, T, Td, prof)
    CAPE, CIN = CAPE.reshape(TIME, LAT, LON), CIN.reshape(TIME, LAT, LON)
    
    # set physical limits
    cape = np.where(CAPE < 8000, CAPE, 8000)
    cin = np.where(CIN > -1400, CIN, -1400)
    
    # Step4:create DataArray 
    ds_cape_cin = xr.Dataset(
        {
            "CAPE": (["time", "rlat", "rlon"], cape),
            "CIN": (["time", "rlat", "rlon"], cin),
        },
        coords={
            "time": array.time[start_ts:end_ts],
            "rlat": array.rlat,
            "rlon": array.rlon,
        }
    )
    return ds_cape_cin

# calculate cape and cin with function
cape_ds_irri_GAR   = calculate_cape_cin(po_irri_GAR_splitt, 0, 720)
cape_ds_noirri_GAR = calculate_cape_cin(po_noirri_GAR_splitt, 0, 720)
cape_ds_irri_SGAR   = calculate_cape_cin(po_irri_SGAR, 0, 720)
cape_ds_noirri_SGAR = calculate_cape_cin(po_noirri_SGAR, 0, 720)

cape_ds_irri_GAR_masked    = cape_ds_irri_GAR.where(irrimask_SGAR > 0)
cape_ds_noirri_GAR_masked  = cape_ds_noirri_GAR.where(irrimask_SGAR > 0)
cape_ds_irri_SGAR_masked   = cape_ds_irri_SGAR.where(irrimask_SGAR > 0)
cape_ds_noirri_SGAR_masked = cape_ds_noirri_SGAR.where(irrimask_SGAR > 0)

# hourly means
cape_hour_irri_GAR   = cape_ds_irri_GAR['CAPE'].groupby('time.hour').mean()
cape_hour_noirri_GAR = cape_ds_noirri_GAR['CAPE'].groupby('time.hour').mean()
cin_hour_irri_GAR   = cape_ds_irri_GAR['CIN'].groupby('time.hour').mean()
cin_hour_noirri_GAR = cape_ds_noirri_GAR['CIN'].groupby('time.hour').mean()

cape_hour_irri_SGAR   = cape_ds_irri_SGAR['CAPE'].groupby('time.hour').mean()
cape_hour_noirri_SGAR  = cape_ds_noirri_SGAR['CAPE'].groupby('time.hour').mean()
cin_hour_irri_SGAR   = cape_ds_irri_SGAR['CIN'].groupby('time.hour').mean()
cin_hour_noirri_SGAR = cape_ds_noirri_SGAR['CIN'].groupby('time.hour').mean()

#hourly diff
diff_hour_cape_GAR  = cape_hour_irri_GAR  - cape_hour_noirri_GAR 
diff_hour_cape_SGAR = cape_hour_irri_SGAR  - cape_hour_noirri_SGAR 
diff_hour_cin_GAR  = abs(cin_hour_irri_GAR)  - abs(cin_hour_noirri_GAR) 
diff_hour_cin_SGAR = abs(cin_hour_irri_SGAR)  - abs(cin_hour_noirri_SGAR) 

# selhour
selhour = 18

cape_selhour_irri_GAR   = cape_hour_irri_GAR.loc[selhour]
cape_selhour_noirri_GAR   = cape_hour_noirri_GAR.loc[selhour]
cin_selhour_irri_GAR   = cin_hour_irri_GAR.loc[selhour]
cin_selhour_noirri_GAR   = cin_hour_noirri_GAR.loc[selhour]

cape_selhour_irri_SGAR   = cape_hour_irri_SGAR.loc[selhour]
cape_selhour_noirri_SGAR   = cape_hour_noirri_SGAR.loc[selhour]
cin_selhour_irri_SGAR   = cin_hour_irri_SGAR.loc[selhour]
cin_selhour_noirri_SGAR   = cin_hour_noirri_SGAR.loc[selhour]

#diff selhour
diff_selhour_cape_GAR = cape_selhour_irri_GAR - cape_selhour_noirri_GAR
diff_selhour_cape_SGAR = cape_selhour_irri_SGAR - cape_selhour_noirri_SGAR
diff_selhour_cin_GAR = abs(cin_selhour_irri_GAR) - abs(cin_selhour_noirri_GAR)
diff_selhour_cin_SGAR = abs(cin_selhour_irri_SGAR) - abs(cin_selhour_noirri_SGAR)


# create 1d values for histograms 
cape_ds_irri_GAR_1d    = cape_ds_irri_GAR_masked['CAPE'].values.flatten()
cape_ds_noirri_GAR_1d  = cape_ds_noirri_GAR_masked['CAPE'].values.flatten()
cape_ds_irri_SGAR_1d   = cape_ds_irri_SGAR_masked['CAPE'].values.flatten()
cape_ds_noirri_SGAR_1d = cape_ds_noirri_SGAR_masked['CAPE'].values.flatten()

cin_ds_irri_GAR_1d    = cape_ds_irri_GAR_masked['CIN'].values.flatten()
cin_ds_noirri_GAR_1d  = cape_ds_noirri_GAR_masked['CIN'].values.flatten()
cin_ds_irri_SGAR_1d   = cape_ds_irri_SGAR_masked['CIN'].values.flatten()
cin_ds_noirri_SGAR_1d = cape_ds_noirri_SGAR_masked['CIN'].values.flatten()


#################### 2. LCL and LFC #######################################

# LCL with nzthermo
# lcl function gives back a LCLs an array with pressure levels. 
# Accoring to https://github.com/Unidata/MetPy/issues/1199 only 
# the lowest array is the LCL that we look for. The others are higher LCL. 
# I will use again nzt as it is faster see comparison here 
# https://github.com/leaver2000/nzthermo-notebooks/blob/master/notebook.ipynb 

def calculate_lcl(array, start_ts, end_ts):
    pressure = array.coords["plev"].to_numpy().astype(np.float32)  # (Pa) 
    temperature = (
        array["T"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    ) # K
    specific_humidity = (
        array["QD"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    )  # kg/kg
    
    P = pressure[::-1]
    Z = len(P)
    # therefore (N, Z) where N == T*Y*X
    T = np.moveaxis(temperature[:, ::-1, :, :], 1, -1).reshape(-1, Z)  # (N, Z)
    
    Td = nzt.dewpoint_from_specific_humidity(
        P[np.newaxis, :],
        np.moveaxis(specific_humidity[:, ::-1, :, :], 1, -1).reshape(-1, Z),
    )  # (K) (N, Z)
    
    lcl_p, lcl_t = nzt.lcl(P, T, Td)
    #print(np.shape(lcl_p))
  # as for CAPE and CIn we have to unstack the array
    TIME,  LAT, LON, PRESSURE =  (temperature.shape[0],)+ temperature.shape[2:]+(temperature.shape[1],)
    LCL_P, LCL_T = lcl_p.reshape( TIME,  LAT, LON, PRESSURE,), lcl_t.reshape(TIME, LAT, LON, PRESSURE,)
    print(np.shape(LCL_P))
    lcl_p0, lcl_t0 = LCL_P[:,:,:,0], LCL_T[:,:,:,0]
    ds_lcl = xr.Dataset(
    {
        "LCL_P": (["time", "rlat", "rlon"], lcl_p0),
        "LCL_T": (["time", "rlat", "rlon"], lcl_t0),
    },
    coords={
        "time": array.time[start_ts: end_ts],
        "rlat": array.rlat,
        "rlon": array.rlon,
    }
    )
    return ds_lcl

# LCL: apply function 
lcl_ds_irri_GAR    = calculate_lcl(po_irri_GAR_splitt, 0, 720)
lcl_ds_noirri_GAR  =  calculate_lcl(po_noirri_GAR_splitt, 0, 720)
lcl_ds_irri_SGAR   = calculate_lcl(po_irri_SGAR, 0, 720)
lcl_ds_noirri_SGAR = calculate_lcl(po_noirri_SGAR, 0, 720)

# I have to switch them to meters (otherwise the pressure increases and that is not very intuitve 
def convert_lcl_to_meters(ds, z0 ):
    # Extract necessary variables
    P = ds["LCL_P"] * units.Pa  # Pressure (Pa)
    T = ds["LCL_T"] * units.K  # Temperature (K)
    P0 = 1075 *units.hPa # as our lowest pressure level
    Z0 = z0 * units.m
    # Constants
    Rd = 287.05 * units.joule / (units.kilogram * units.kelvin)  # Gas constant for dry air
    g = 9.81 * units.meter / (units.second ** 2)  # Gravity acceleration
    # hypsometric equation, p0 has to be surface pressure/lowest level
    Z = Z0 + ((Rd / g) * T * np.log(P0 / P))
    return Z
    
z0_GAR = po_irri_GAR_hor_split.FIB
z0_SGAR = po_irri_SGAR_hor.FIB

lcl_irri_GAR_m    = convert_lcl_to_meters(lcl_ds_irri_GAR, z0_GAR)
lcl_noirri_GAR_m  = convert_lcl_to_meters(lcl_ds_noirri_GAR, z0_GAR)
lcl_irri_SGAR_m   = convert_lcl_to_meters(lcl_ds_irri_SGAR, z0_SGAR)
lcl_noirri_SGAR_m = convert_lcl_to_meters(lcl_ds_noirri_SGAR, z0_SGAR)

lcl_irri_GAR_m_masked    = lcl_irri_GAR_m.where(irrimask_SGAR>0)
lcl_noirri_GAR_m_masked  = lcl_noirri_GAR_m.where(irrimask_SGAR>0)
lcl_irri_SGAR_m_masked   = lcl_irri_SGAR_m.where(irrimask_SGAR>0)
lcl_noirri_SGAR_m_masked = lcl_noirri_SGAR_m.where(irrimask_SGAR>0)

# create mean value for spatial plot 
lcl_hour_irri_GAR     = lcl_irri_GAR_m.groupby('time.hour').mean()
lcl_hour_noirri_GAR   = lcl_noirri_GAR_m.groupby('time.hour').mean()
lcl_hour_irri_SGAR    = lcl_irri_SGAR_m.groupby('time.hour').mean()
lcl_hour_noirri_SGAR  = lcl_noirri_SGAR_m.groupby('time.hour').mean()

# selhour
selhour = 18
lcl_selhour_irri_GAR     = lcl_hour_irri_GAR.loc[:,:,selhour]
lcl_selhour_noirri_GAR   = lcl_hour_noirri_GAR.loc[:,:,selhour]
lcl_selhour_irri_SGAR    = lcl_hour_irri_SGAR.loc[:,:,selhour]
lcl_selhour_noirri_SGAR  = lcl_hour_noirri_SGAR.loc[:,:,selhour]

#diff
diff_selhour_lcl_GAR  = lcl_selhour_irri_GAR - lcl_selhour_noirri_GAR
diff_selhour_lcl_SGAR = lcl_selhour_irri_SGAR - lcl_selhour_noirri_SGAR

# create 1d values for histogram
lcl_ds_irri_GAR_1d    = lcl_irri_GAR_m_masked.values.flatten()
lcl_ds_noirri_GAR_1d  = lcl_noirri_GAR_m_masked.values.flatten()
lcl_ds_irri_SGAR_1d   = lcl_irri_SGAR_m_masked.values.flatten()
lcl_ds_noirri_SGAR_1d = lcl_noirri_SGAR_m_masked.values.flatten()

# LFC with nzthermo, adopted from LCL
def calculate_lfc(array, start_ts, end_ts):
    
    pressure = array.coords["plev"].to_numpy().astype(np.float32)  # (Pa) 
    temperature = (
        array["T"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    ) # K
    specific_humidity = (
        array["QD"].isel(time=slice(start_ts, end_ts)).to_numpy().astype(np.float32)
    )  # kg/kg
    
    P = pressure[::-1]
    Z = len(P)
    # therefore (N, Z) where N == T*Y*X
    T = np.moveaxis(temperature[:, ::-1, :, :], 1, -1).reshape(-1, Z)  # (N, Z)
    Td = nzt.dewpoint_from_specific_humidity(
        P[np.newaxis, :],
        np.moveaxis(specific_humidity[:, ::-1, :, :], 1, -1).reshape(-1, Z),
    )  # (K) (N, Z)
    
    lfc_p, lfc_t = nzt.lfc(P, T, Td)
    # as for CAPE and CIn we have to unstack the array
    TIME,  LAT, LON =  (temperature.shape[0],)+ temperature.shape[2:]
    LFC_P, LFC_T = lfc_p.reshape( TIME,  LAT, LON,), lfc_t.reshape(TIME, LAT, LON)
    
    ds_lfc = xr.Dataset(
    {
        "LFC_P": (["time", "rlat", "rlon"], LFC_P),
        "LFC_T": (["time", "rlat", "rlon"], LFC_T),
    },
    coords={
        "time": array.time[start_ts: end_ts],
        "rlat": array.rlat,
        "rlon": array.rlon,
    }
    )
    return ds_lfc

# calcualte LFC and apply function
lfc_ds_irri_GAR   = calculate_lfc(po_irri_GAR_splitt, 0, 720)
lfc_ds_noirri_GAR = calculate_lfc(po_noirri_GAR_splitt, 0, 720)

lfc_ds_irri_SGAR   = calculate_lfc(po_irri_SGAR, 0, 720)
lfc_ds_noirri_SGAR = calculate_lfc(po_noirri_SGAR, 0, 720)

# I have to switch them to meters (otherwise the pressure increases and that is not very intuitve 
def convert_lfc_to_meters(ds, z0 ):
    # Extract necessary variables
    P = ds["LFC_P"] * units.Pa  # Pressure (Pa)
    T = ds["LFC_T"] * units.K  # Temperature (K)
    P0 = 1075 *units.hPa # as our lowest pressure level
    Z0 = z0 * units.m
    # Constants
    Rd = 287.05 * units.joule / (units.kilogram * units.kelvin)  # Gas constant for dry air
    g = 9.81 * units.meter / (units.second ** 2)  # Gravity acceleration
    # hypsometric equation, p0 has to be surface pressure/lowest level
    Z = Z0 + ((Rd / g) * T * np.log(P0 / P))
    return Z
    
z0_GAR  = po_irri_GAR_hor_split.FIB
z0_SGAR = po_irri_SGAR_hor.FIB

lfc_irri_GAR_m    = convert_lfc_to_meters(lfc_ds_irri_GAR, z0_GAR)
lfc_noirri_GAR_m  = convert_lfc_to_meters(lfc_ds_noirri_GAR, z0_GAR)
lfc_irri_SGAR_m   = convert_lfc_to_meters(lfc_ds_irri_SGAR, z0_SGAR)
lfc_noirri_SGAR_m = convert_lfc_to_meters(lfc_ds_noirri_SGAR, z0_SGAR)

lfc_irri_GAR_m_masked    = lfc_irri_GAR_m.where(irrimask_SGAR>0)
lfc_noirri_GAR_m_masked  = lfc_noirri_GAR_m.where(irrimask_SGAR>0)
lfc_irri_SGAR_m_masked   = lfc_irri_SGAR_m.where(irrimask_SGAR>0)
lfc_noirri_SGAR_m_masked = lfc_noirri_SGAR_m.where(irrimask_SGAR>0)


# create mean value for spatial plot 
lfc_hour_irri_GAR     = lfc_irri_GAR_m.groupby('time.hour').mean()
lfc_hour_noirri_GAR   = lfc_noirri_GAR_m.groupby('time.hour').mean()
lfc_hour_irri_SGAR    = lfc_irri_SGAR_m.groupby('time.hour').mean()
lfc_hour_noirri_SGAR  = lfc_noirri_SGAR_m.groupby('time.hour').mean()

# selhour
selhour = 18
lfc_selhour_irri_GAR     = lfc_hour_irri_GAR.loc[:,:,selhour]
lfc_selhour_noirri_GAR   = lfc_hour_noirri_GAR.loc[:,:,selhour]
lfc_selhour_irri_SGAR    = lfc_hour_irri_SGAR.loc[:,:,selhour]
lfc_selhour_noirri_SGAR  = lfc_hour_noirri_SGAR.loc[:,:,selhour]

#diff
diff_selhour_lfc_GAR  = lfc_selhour_irri_GAR - lfc_selhour_noirri_GAR
diff_selhour_lfc_SGAR = lfc_selhour_irri_SGAR - lfc_selhour_noirri_SGAR


# create 1d values for histogram
lfc_ds_irri_GAR_1d    = lfc_irri_GAR_m_masked.values.flatten()
lfc_ds_noirri_GAR_1d  = lfc_noirri_GAR_m_masked.values.flatten()
lfc_ds_irri_SGAR_1d   = lfc_irri_SGAR_m_masked.values.flatten()
lfc_ds_noirri_SGAR_1d = lfc_noirri_SGAR_m_masked.values.flatten()


##################### PLOT ###########################################

def plot_density_histogram(fig, ax2, data1_irri, data1_noirri, data2_irri, data2_noirri, bins, legend, params):  
    ## lclitation    

    plt.rcParams.update(params)

    hist_plot = ax2.hist([\
              np.clip(data1_irri, bins[0], bins[-1]), \
              np.clip(data1_noirri, bins[0], bins[-1]),\
              np.clip(data2_irri, bins[0], bins[-1]), \
              np.clip(data2_noirri, bins[0], bins[-1])],\
              bins=bins, histtype='bar', density = True)
      
        # get the current axis
    ax2 = plt.gca()
    # Shink current axis by 20%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    colors = ['cornflowerblue', 'darkorange', 'white', 'white','dimgrey']
    hatches = [None,None, '///','///']
    hatchcolors = [None, None, 'cornflowerblue', 'darkorange']

    iter_colors = np.repeat(colors,len(bins)-1)
    iter_hatches = np.tile(np.repeat(hatches, len(bins)-1),2)
    iter_hatchcolors = np.repeat(hatchcolors,len(bins)-1)

    for patch,color, hatch, hatchcolor in zip(ax2.patches, iter_colors, iter_hatches, iter_hatchcolors):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)
        patch.set_edgecolor(hatchcolor)
    if legend == True: 
        # Add legends:
        sim = ['irrigated', 'not irrigated']
        res = ['0.11°', '0.0275°']
        color_legend = ['cornflowerblue', 'darkorange']
        # get the current axis
        sim_legend = ax2.legend(
            [Line2D([0], [0], color=color, lw=4) for color in color_legend],
            sim, title='simulation', 
            loc = 'best',bbox_to_anchor=(1.26, 0.999), borderaxespad=0, 
            bbox_transform=ax2.transAxes,  
            handlelength= 1.3,  # Shrink line length in legend
            handletextpad=0.5,  # Reduce space between line and text
            #columnspacing=0.8,  # Reduce spacing between columns (if applicable)
           # labelspacing=0.3,  # Reduce vertical space between labels
        )

        res_legend = ax2.legend(
            [Patch(hatch=hatch, facecolor='white', edgecolor = 'black') for hatch in hatches[1::2]],
            res, bbox_to_anchor=(0.99, 0.48), bbox_transform=ax2.transAxes, 
            title='resolution', labelspacing=.65,
            handletextpad=0.5,  # Reduce space between line and text
        )

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
#########################################################
############################# PLOT FIGURE################

gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1.5], height_ratios=[1.2, 1.2, 1.2, 1.2])  # Last column is wider

fig = plt.figure(figsize=(18,13))

params = {
    "legend.fontsize": 10,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(params)

cax1 = fig.add_axes([0.15, 0.73, 0.3, 0.02])
cax2 = fig.add_axes([0.15, 0.513, 0.3, 0.02])
cax3 = fig.add_axes([0.15, 0.3, 0.3, 0.02])
cax4 = fig.add_axes([0.15, 0.08, 0.3, 0.02])

cape_levels = np.arange(-1000, 1100, 100)
cape_ticks = np.arange(-1000, 1500, 500)

cin_levels = np.arange(-100,110, 10)
cin_ticks = np.arange(-100,150, 50)

lcl_levels = np.arange(-1000, 1100, 100)
lcl_ticks = np.arange(-1000, 1500, 500)

lfc_levels = np.arange(-1000, 1100, 100)
lfc_ticks = np.arange(-1000, 1500, 500)


# CAPE
ax1 = fig.add_subplot(gs[0, 0], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_cape_GAR, ax1, cax1, '', ' Δ CAPE [Jkg$^{-1}$]', 'RdBu_r', cape_levels, cape_ticks, 'both', 'horizontal')
ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=14)

ax2 = fig.add_subplot(gs[0, 1], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_cape_SGAR, ax2, cax1, '', ' Δ CAPE [Jkg$^{-1}$]', 'RdBu_r', cape_levels, cape_ticks, 'both', 'horizontal')
ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.0, 1.08, '(b)', transform=ax2.transAxes, fontsize=14)

ax5 = fig.add_subplot(gs[0, 2])
steps = 500
max_val = 4500
max_tick = max_val - steps
bins = np.arange(0,max_val, steps)
plot_density_histogram(fig, ax5, cape_ds_irri_GAR_1d, cape_ds_noirri_GAR_1d, cape_ds_irri_SGAR_1d, cape_ds_noirri_SGAR_1d, bins, True, params)  
ax5.set_xlabel('CAPE [Jkg$^{-1}$]')
labels = [item.get_text() for item in ax5.get_xticklabels()]
labels[-1] = '≥ '+str(int(max_tick))
ax5.set_xticklabels(labels)
ax5.set_ylim(10**(-6))
ax5.text(0.0, 1.08, '(c)', transform=ax5.transAxes, fontsize=14)

# CIN
ax3 = fig.add_subplot(gs[1, 0], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_cin_GAR, ax3, cax2, '', 'Δ CIN [Jkg$^{-1}$]', 'RdBu_r', cin_levels, cin_ticks, 'both', 'horizontal')
ax3.text(0.035, 0.964, '0.11°', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax3.text(0.0, 1.08, '(d)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(gs[1,1], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_cin_SGAR, ax4, cax2, '', 'Δ CIN [Jkg$^{-1}$]', 'RdBu_r', cin_levels, cin_ticks, 'both', 'horizontal')
ax4.text(0.035, 0.964, '0.0275°', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax4.text(0.0, 1.08, '(e)', transform=ax4.transAxes, fontsize=14)

ax6 = fig.add_subplot(gs[1,2])
steps = 100
min_tick = -800
bins = np.arange(-800, 100, steps)
plot_density_histogram(fig, ax6, cin_ds_irri_GAR_1d, cin_ds_noirri_GAR_1d, cin_ds_irri_SGAR_1d, cin_ds_noirri_SGAR_1d, bins, False, params)  
ax6.set_xlabel('CIN [Jkg$^{-1}$]')
tick_positions = bins[0:]  # Shift to the right edge
tick_labels = [f"{int(b)}" for b in tick_positions]  # Convert bin edges to labels
tick_labels[0] = '≤ '+str(int(min_tick))
ax6.set_xticks(tick_positions[::2])  
ax6.set_xticklabels(tick_labels[::2], ha="center")  # Align left for clarity
ax6.set_ylim(10**(-6))
ax6.text(0.0, 1.08, '(f)', transform=ax6.transAxes, fontsize=14)


# LCL
ax7 = fig.add_subplot(gs[2, 0], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_lcl_GAR, ax7, cax3, '', ' Δ LCL [m]', 'RdBu_r', lcl_levels, lcl_ticks, 'both', 'horizontal')
ax7.text(0.035, 0.964, '0.11°', transform=ax7.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax7.text(0.0, 1.08, '(g)', transform=ax7.transAxes, fontsize=14)

ax8 = fig.add_subplot(gs[2, 1], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_lcl_SGAR, ax8, cax3, '', ' Δ LCL [m]', 'RdBu_r', lcl_levels, lcl_ticks, 'both', 'horizontal')
ax8.text(0.035, 0.964, '0.0275°', transform=ax8.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax8.text(0.0, 1.08, '(h)', transform=ax8.transAxes, fontsize=14)

ax9 = fig.add_subplot(gs[2,2])
steps = 500
max_val = 4500
max_tick = max_val - steps
bins = np.arange(500,max_val, steps)
plot_density_histogram(fig, ax9, lcl_ds_irri_GAR_1d, lcl_ds_noirri_GAR_1d, lcl_ds_irri_SGAR_1d, lcl_ds_noirri_SGAR_1d, bins, False, params)  
ax9.set_xlabel('LCL [m]')
labels = [item.get_text() for item in ax9.get_xticklabels()]
labels[-1] = '≥ '+str(int(max_tick))
ax9.set_xticklabels(labels)
ax9.set_ylim(10**(-6))
ax9.text(0.0, 1.08, '(i)', transform=ax9.transAxes, fontsize=14)

# LFC
ax10 = fig.add_subplot(gs[3, 0], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_lfc_GAR, ax10, cax4, '', ' Δ LFC [m]', 'RdBu_r', lfc_levels, lfc_ticks, 'both', 'horizontal')
ax10.text(0.035, 0.964, '0.11°', transform=ax10.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax10.text(0.0, 1.08, '(j)', transform=ax10.transAxes, fontsize=14)

ax11 = fig.add_subplot(gs[3, 1], projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_selhour_lfc_SGAR, ax11, cax4, '', ' Δ LFC [m]', 'RdBu_r', lfc_levels, lfc_ticks, 'both', 'horizontal')
ax11.text(0.035, 0.964, '0.0275°', transform=ax11.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax11.text(0.0, 1.08, '(k)', transform=ax11.transAxes, fontsize=14)

ax12 = fig.add_subplot(gs[3, 2])
steps = 500
max_val = 4500
max_tick = max_val - steps
bins = np.arange(500,max_val, steps)
plot_density_histogram(fig, ax12, lfc_ds_irri_GAR_1d, lfc_ds_noirri_GAR_1d, lfc_ds_irri_SGAR_1d, lfc_ds_noirri_SGAR_1d, bins, False, params)  
ax12.set_xlabel('LFC [m]')
labels = [item.get_text() for item in ax12.get_xticklabels()]
labels[-1] = '≥ '+str(int(max_tick))
ax12.set_xticklabels(labels)
ax12.set_ylim(10**(-6))
ax12.text(0.0, 1.08, '(l)', transform=ax12.transAxes, fontsize=14)

#plt.tight_layout()
plt.subplots_adjust(hspace=0.8, wspace=0.2, right=0.8)
plt.savefig(str(dir_out)+'/cape_cin_lcl_lfc_subplot.png',dpi=300, bbox_inches='tight')

############################### PLOT ABS VALUES #######################################

fig = plt.figure(figsize=(15,5))

cax1 = fig.add_axes([0.32, 0.52, 0.37, 0.03])
levels_lcl = np.arange(0,5200,200)
ticks_lcl = np.arange(0,5500,500)

cax2 = fig.add_axes([0.32, 0.07, 0.37, 0.03])
levels_lfc = np.arange(0,5250,250)
ticks_lfc = np.arange(0,5500,500)


ax1 = fig.add_subplot(2, 4, 1, projection=rotated_pole)
rotplot = plot_rotvar(fig, lcl_selhour_irri_GAR, ax1, cax1, 'mm', 'LCL [m]', 'plasma', levels_lcl, ticks_lcl, 'max', 'horizontal')
ax1.set_title('')
ax1.text(0.035, 0.964, '0.11° irri', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=14)

ax2 = fig.add_subplot(2, 4, 2, projection=rotated_pole)
rotplot = plot_rotvar(fig, lcl_selhour_noirri_GAR, ax2, cax1, 'mm', 'LCL [m]', 'plasma', levels_lcl, ticks_lcl, 'max', 'horizontal')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.11° noirri', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.0, 1.08, '(b)', transform=ax2.transAxes, fontsize=14)

ax3 = fig.add_subplot(2, 4, 3, projection=rotated_pole)
rotplot = plot_rotvar(fig, lcl_selhour_irri_SGAR, ax3, cax1, 'mm', 'LCL [m]', 'plasma', levels_lcl, ticks_lcl, 'max', 'horizontal')
ax3.set_title('')
ax3.text(0.035, 0.964, '0.0275° irri', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax3.text(0.0, 1.08, '(c)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(2, 4, 4, projection=rotated_pole)
rotplot = plot_rotvar(fig, lcl_selhour_noirri_SGAR, ax4, cax1, 'mm', 'LCL [m]', 'plasma', levels_lcl, ticks_lcl, 'max', 'horizontal')
ax4.set_title('')
ax4.text(0.035, 0.964, '0.0275° noirri', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax4.text(0.0, 1.08, '(d)', transform=ax4.transAxes, fontsize=14)

# lfcspeed 10 m
ax5 = fig.add_subplot(2, 4, 5, projection=rotated_pole)
rotplot = plot_rotvar(fig, lfc_selhour_irri_GAR, ax5, cax2, 'mm', 'LFC [m]', 'plasma', levels_lfc, ticks_lfc, 'max', 'horizontal')
ax5.set_title('')
ax5.text(0.035, 0.964, '0.11° irri', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax5.text(0.0, 1.08, '(e)', transform=ax5.transAxes, fontsize=14)


ax6 = fig.add_subplot(2, 4, 6, projection=rotated_pole)
rotplot = plot_rotvar(fig, lfc_selhour_noirri_GAR, ax6, cax2, 'mm', 'LFC  [m]', 'plasma', levels_lfc, ticks_lfc, 'max', 'horizontal')
ax6.set_title('')
ax6.text(0.035, 0.964, '0.11° noirri', transform=ax6.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax6.text(0.0, 1.08, '(f)', transform=ax6.transAxes, fontsize=14)

ax7 = fig.add_subplot(2, 4, 7, projection=rotated_pole)
rotplot = plot_rotvar(fig, lfc_selhour_irri_SGAR, ax7, cax2, 'mm', 'LFC [m]', 'plasma', levels_lfc, ticks_lfc, 'max', 'horizontal')
ax7.set_title('')
ax7.text(0.035, 0.964, '0.0275 irri°', transform=ax7.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax7.text(0.0, 1.08, '(g)', transform=ax7.transAxes, fontsize=14)


ax8 = fig.add_subplot(2, 4, 8, projection=rotated_pole)
rotplot = plot_rotvar(fig, lfc_selhour_noirri_SGAR, ax8, cax2, 'mm', 'LFC [m]', 'plasma', levels_lfc, ticks_lfc, 'max', 'horizontal')
ax8.set_title('')
ax8.text(0.035, 0.964, '0.0275 noirri°', transform=ax8.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax8.text(0.0, 1.08, '(h)', transform=ax8.transAxes, fontsize=14)


fig.subplots_adjust(hspace = 0.5)
plt.savefig(str(dir_out)+'/lcl_lfc_abs.png',dpi=300, bbox_inches='tight')





