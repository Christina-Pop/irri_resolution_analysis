#!/usr/bin/env python
# coding: utf-8

import pyremo as pr
from pyremo.physics import pressure
from pyremo.prsint import pressure_interpolation
import os
import metpy.calc
from metpy.units import units

print('working directory',os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd 

# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
# print("Output directory is: ", dir_out)
############################################ READ E-FILES#####################


year = 2017
month = 0
#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"

varlist_SGAR = [ "GHPBL"]
var_num_list_SGAR = [ "271"]

varlist_GAR =       ["GHPBL"]
var_num_list_GAR = [ "271"]


# adapt your paths where the data is stored.
# Please also check the functions in functions_reading_files.py. 
data_path_SGAR_noirri = "/noirri0275"
data_path_SGAR_irri = "/irri0275"
data_path_GAR_noirri = "/noirri11"
data_path_GAR_irri = "/irri11"

# SGAR 
for var_SGAR, var_num_SGAR in zip(varlist_SGAR, var_num_list_SGAR):
    single_var_data_SGAR_irri = read_efiles_in_chunks(data_path_SGAR_irri, var_SGAR, var_num_SGAR, exp_number_irri_SGAR, year, month, 100, 1, 100, 100)
    single_var_data_SGAR_noirri = read_efiles_in_chunks(data_path_SGAR_noirri, var_SGAR, var_num_SGAR, exp_number_noirri_SGAR, year, month, 100, 1, 100, 100)
   
    if var_SGAR == varlist_SGAR[0]:
        ds_var_irri_SGAR = single_var_data_SGAR_irri
        ds_var_noirri_SGAR = single_var_data_SGAR_noirri
    else:
        ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, single_var_data_SGAR_irri])
        ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, single_var_data_SGAR_noirri])
# GAR
for var_GAR, var_num_GAR in zip(varlist_GAR, var_num_list_GAR):
    single_var_data_GAR_irri = read_efiles_in_chunks(data_path_GAR_irri, var_GAR, var_num_GAR, exp_number_irri_GAR, year, month, 100, 1, 100, 100)
    single_var_data_GAR_noirri = read_efiles_in_chunks(data_path_GAR_noirri, var_GAR, var_num_GAR, exp_number_noirri_GAR, year, month, 100, 1, 100, 100)
   
    if var_GAR == varlist_GAR[0]:
        ds_var_irri_GAR = single_var_data_GAR_irri
        ds_var_noirri_GAR = single_var_data_GAR_noirri
    else:
        ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, single_var_data_GAR_irri])
        ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, single_var_data_GAR_noirri])


# adding irrifrac 
irrifrac_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_SGAR)+'e_c743_201706.nc'
irrifrac_data_SGAR = xr.open_dataset(irrifrac_file_SGAR)
irrifrac_SGAR = irrifrac_data_SGAR.IRRIFRAC[0]

irrifrac_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_GAR)+'e_c743_201706.nc'
irrifrac_data_GAR = xr.open_dataset(irrifrac_file_GAR)
irrifrac_GAR = irrifrac_data_GAR.IRRIFRAC[0]

lsm_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/BLA/e'+str(exp_number_noirri_SGAR)+'e_c172_201706.nc'
lsm_data_SGAR = xr.open_dataset(lsm_file_SGAR)
lsm_SGAR = lsm_data_SGAR.BLA[0]

lsm_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/BLA/e'+str(exp_number_noirri_GAR)+'e_c172_201706.nc'
lsm_data_GAR = xr.open_dataset(lsm_file_GAR)
lsm_GAR = lsm_data_GAR.BLA[0]
#

topo_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/FIB/e'+str(exp_number_noirri_SGAR)+'e_c129_201706.nc'
topo_data_SGAR = xr.open_dataset(topo_file_SGAR)
topo_SGAR = topo_data_SGAR.FIB[0]

topo_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/FIB/e'+str(exp_number_noirri_GAR)+'e_c129_201706.nc'
topo_data_GAR = xr.open_dataset(topo_file_GAR)
topo_GAR = topo_data_GAR.FIB[0]

#
ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR, lsm_SGAR,topo_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR, lsm_SGAR, topo_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR, lsm_GAR, topo_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR, lsm_GAR, topo_GAR])

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, topo_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, topo_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, topo_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, topo_GAR])

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
po_irri_GAR   = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108), rlat_staggered=slice(50, 72), rlon_staggered=slice(70, 108))
po_noirri_GAR = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108), rlat_staggered=slice(50, 72), rlon_staggered=slice(70, 108))

po_noirri_SGAR = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR)
po_irri_SGAR   = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR)
po_noirri_SGAR = po_noirri_SGAR.isel(rlon_staggered = slice(78,78+148),rlat_staggered = slice(110,110+84))
po_irri_SGAR   = po_irri_SGAR.isel(rlon_staggered = slice(78,78+148),rlat_staggered = slice(110,110+84))

 # select months
po_irri_GAR    = po_irri_GAR.sel(time=po_irri_GAR.time.dt.month.isin([6]))
po_noirri_GAR   = po_noirri_GAR.sel(time=po_noirri_GAR.time.dt.month.isin([6]))
po_irri_SGAR    = po_irri_SGAR.sel(time=po_irri_SGAR.time.dt.month.isin([6]))
po_noirri_SGAR   = po_noirri_SGAR.sel(time=po_noirri_SGAR.time.dt.month.isin([6]))

# split 0.11 in 0.0275 
po_noirri_GAR_split = po_noirri_GAR.interp_like(
    po_noirri_SGAR,
    method='nearest')
po_irri_GAR_split = po_irri_GAR.interp_like(
    po_irri_SGAR,
    method='nearest')

# PBL calculation: diurnal cycle for PBL 
pbl_GAR_irri_day    = po_irri_GAR_split.GHPBL.groupby('time.hour').mean('time')
pbl_GAR_noirri_day  = po_noirri_GAR_split.GHPBL.groupby('time.hour').mean('time')
pbl_SGAR_irri_day   = po_irri_SGAR.GHPBL.groupby('time.hour').mean('time')
pbl_SGAR_noirri_day = po_noirri_SGAR.GHPBL.groupby('time.hour').mean('time')

pbl_GAR_irri_day_sel   = (pbl_GAR_irri_day.where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])
pbl_GAR_noirri_day_sel = (pbl_GAR_noirri_day.where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])

pbl_SGAR_irri_day_sel   = (pbl_SGAR_irri_day.where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])
pbl_SGAR_noirri_day_sel = (pbl_SGAR_noirri_day.where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])


def convert_gpm2m(gpm):
    g0 = 9.80665  # Gravity (m/s²)
    Re = 6371000  # Earth radius in meters
    return (Re * gpm) / (Re - gpm)

#conversion necessary
pblh_irri_GAR_meter       = convert_gpm2m(pbl_GAR_irri_day_sel.__xarray_dataarray_variable__ )
pblh_noirri_GAR_meter     = convert_gpm2m(pbl_GAR_noirri_day_sel.__xarray_dataarray_variable__ )
pblh_irri_SGAR_meter      = convert_gpm2m(pbl_SGAR_irri_day_sel.__xarray_dataarray_variable__ )
pblh_noirri_SGAR_meter    = convert_gpm2m(pbl_SGAR_noirri_day_sel.__xarray_dataarray_variable__ )

def convert_height_to_pressure_da(height_da):
    # Constants
    P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
    H = 8434.5    # Scale height in meters for Earth's atmosphere

    # Apply the barometric formula element-wise
    pressure_da = P0 * np.exp(-height_da / H)

    # Preserve metadata
    pressure_da.attrs = height_da.attrs.copy()
    pressure_da.attrs['units'] = 'hPa'
    pressure_da.name = 'pressure_height'
    return pressure_da

pblh_irri_GAR_pressure        = convert_height_to_pressure_da(pblh_irri_GAR_meter )*100
pblh_noirri_GAR_pressure      = convert_height_to_pressure_da(pblh_noirri_GAR_meter )*100
pblh_irri_SGAR_pressure       = convert_height_to_pressure_da(pblh_irri_SGAR_meter )*100
pblh_noirri_SGAR_pressure     = convert_height_to_pressure_da(pblh_noirri_SGAR_meter )*100

############################################ READ P-FILES#####################

# adjust the paths where you stored the data
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

############################ CALCULATIONS ##########################################
# the mask mask_day.nc and mask_night.nc are created with the script Fig8_Fig9_temperature_range.py
# use values which change their sign in Figure 9
mask_day = xr.open_dataset('mask_day.nc')
mask_night = xr.open_dataset('mask_night.nc')

# check the timing of irri signal in higher levels
# temperature
diff_GAR  = po_irri_GAR_splitt.T - po_noirri_GAR_splitt.T
diff_SGAR = po_irri_SGAR.T - po_noirri_SGAR.T
diff_GAR_day  = (diff_GAR.groupby('time.hour').mean('time').where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])
diff_SGAR_day = (diff_SGAR.groupby('time.hour').mean('time').where((mask_night>0.5)| (mask_day>0.5))).mean(['rlat','rlon'])

############################# PLOT PROFILE ############################################
diff_GAR  = po_irri_GAR_splitt.T - po_noirri_GAR_splitt.T
diff_SGAR = po_irri_SGAR.T - po_noirri_SGAR.T
diff_GAR_day  = (diff_GAR.groupby('time.hour').mean('time').where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon'])
diff_SGAR_day = (diff_SGAR.groupby('time.hour').mean('time').where((mask_night>0.5)| (mask_day>0.5))).mean(['rlat','rlon'])


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

cax = fig.add_axes([1.05, 0.12, 0.02, 0.8])


#GAR 
levels = list(filter(lambda num: num != 0, np.arange(-2, 2.2, 0.2)))
plot1 = diff_GAR_day.__xarray_dataarray_variable__.T.plot(ax=ax1, levels=levels, add_colorbar = False)
pblh_irri_GAR_pressure.plot(ax = ax1, zorder = 3, color = 'black', label = 'irrigated')
pblh_noirri_GAR_pressure.plot(ax = ax1, zorder = 3, color = 'black', label = 'non-irrigated', linestyle = '--')  
ax1.invert_yaxis()
ax1.set_xlabel("Hour")
ax1.set_ylabel("pressure [Pa]")
ax1.set_ylim(100500, 70000)
ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.set_title('')
ax1.legend()

#SGAR
plot2 = diff_SGAR_day.__xarray_dataarray_variable__.T.plot(ax=ax2, levels=levels, add_colorbar = False, zorder = 2 )
pblh_irri_SGAR_pressure.plot(ax = ax2, zorder = 3, color = 'black', label = 'irrigated')
pblh_noirri_SGAR_pressure.plot(ax = ax2, zorder = 3, color = 'black', label = 'non-irrigated', linestyle = '--')  
ax2.invert_yaxis()
ax2.set_xlabel("Hour")
ax2.set_ylabel("")
ax2.set_ylim(100500, 70000)
ax2.set_ylabel('')
ax2.set_title('')

ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
cbar = fig.colorbar(
           plot2, cax=cax, label='Δ temperature [K]')
cbar.outline.set_visible(False)
ax2.legend()

plt.tight_layout()
plt.savefig(dir_out+'/vertical_cooling_pbl_hours_appendix_new.png', dpi=300, bbox_inches='tight')



