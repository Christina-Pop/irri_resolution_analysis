#!/usr/bin/env python
# coding: utf-8


import pyremo as pr
import metpy.calc
from metpy.units import units
from metpy.calc import relative_humidity_from_specific_humidity, mixing_ratio_from_specific_humidity
import os

print('working directory',os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from functions_correcting_time import correct_timedim, correct_timedim_mfiles
from functions_plotting import plot_rotvar, plot_rotvar_adjust_cbar, rotated_pole
from functions_reading_files import read_efiles, read_mfiles, read_efiles_in_chunks 

# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

########################### READ DATA ########################

year = 2017
month = 0
#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"

varlist_SGAR = [ "T", "APRL","QD", "U10", "V10", "WIND10", "GHPBL", "TKE"]
var_num_list_SGAR = [ "130", "142", "133", "165", "166", "171", "271", "224"]

varlist_GAR =       [ "T","APRL","APRC","QD", "U10", "V10", "WIND10", "GHPBL", "TKE"]
var_num_list_GAR = [ "130", "142","143","133", "165", "166", "171", "271", "224"]


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


################################################### CALCULATIONS ##############################################

# precipitation 
precip_lim = 1

po_irri_GAR_split['PRECIP'] = po_irri_GAR_split.APRL+po_irri_GAR_split.APRC
po_noirri_GAR_split['PRECIP'] = po_noirri_GAR_split.APRL+po_noirri_GAR_split.APRC
precip_diff_GAR = po_irri_GAR_split['PRECIP']-po_noirri_GAR_split['PRECIP']

precip_month_irri_GAR = po_irri_GAR_split['PRECIP'].resample(time = '1M').sum().squeeze('time')
precip_month_irri_GAR = precip_month_irri_GAR.where(precip_month_irri_GAR > precip_lim)
precip_month_noirri_GAR = po_noirri_GAR_split['PRECIP'].resample(time = '1M').sum().squeeze('time')
precip_month_noirri_GAR = precip_month_noirri_GAR.where(precip_month_noirri_GAR > precip_lim)


po_irri_SGAR['PRECIP'] = po_irri_SGAR.APRL
po_noirri_SGAR['PRECIP'] = po_noirri_SGAR.APRL
precip_diff_SGAR = po_irri_SGAR['PRECIP']-po_noirri_SGAR['PRECIP']

precip_month_irri_SGAR = po_irri_SGAR['PRECIP'].resample(time = '1M').sum().squeeze('time')
precip_month_irri_SGAR = precip_month_irri_SGAR.where(precip_month_irri_SGAR > precip_lim)

precip_month_noirri_SGAR = po_noirri_SGAR['PRECIP'].resample(time ='1M').sum().squeeze('time')
precip_month_noirri_SGAR = precip_month_noirri_SGAR.where(precip_month_noirri_SGAR > precip_lim)

precip_month_diff_GAR = precip_month_irri_GAR - precip_month_noirri_GAR
precip_month_diff_SGAR = precip_month_irri_SGAR - precip_month_noirri_SGAR

irrifrac_GAR  = po_irri_GAR_split.IRRIFRAC
irrifrac_SGAR = po_irri_SGAR.IRRIFRAC


precip_percent_GAR = precip_month_diff_GAR.where(irrifrac_GAR>0).sum(['rlat','rlon'])/precip_month_noirri_GAR.where(irrifrac_GAR>0).sum(['rlat','rlon'])
precip_percent_SGAR = precip_month_diff_SGAR.where(irrifrac_SGAR>0).sum(['rlat','rlon'])/precip_month_noirri_SGAR.where(irrifrac_SGAR>0).sum(['rlat','rlon'])
print('precipitation change GAR:', precip_percent_GAR.values )
print('precipitation change SGAR:', precip_percent_SGAR.values )


precip_percent_GAR = precip_month_diff_GAR.where(irrifrac_GAR>0.7).sum(['rlat','rlon'])/precip_month_noirri_GAR.where(irrifrac_GAR>0.7).sum(['rlat','rlon'])
precip_percent_SGAR = precip_month_diff_SGAR.where(irrifrac_SGAR>0.7).sum(['rlat','rlon'])/precip_month_noirri_SGAR.where(irrifrac_SGAR>0.7).sum(['rlat','rlon'])
print('precipitation change GAR:', precip_percent_GAR.values )
print('precipitation change SGAR:', precip_percent_SGAR.values )


precip_percent_GAR = precip_month_irri_GAR.where(irrifrac_GAR>0).sum(['rlat','rlon'])/precip_month_noirri_GAR.where(irrifrac_GAR>0).sum(['rlat','rlon'])
precip_percent_SGAR = precip_month_irri_SGAR.where(irrifrac_SGAR>0).sum(['rlat','rlon'])/precip_month_noirri_SGAR.where(irrifrac_SGAR>0).sum(['rlat','rlon'])
print('precipitation change GAR:', precip_percent_GAR.values )
print('precipitation change SGAR:', precip_percent_SGAR.values )

# wind speed calculation 10 m 

def calc_uv10_mean(array):
    u10_month = array['U10'].resample(time = '1M').mean().squeeze(['time','height10m'])
    v10_month = array['V10'].resample(time = '1M').mean().squeeze(['time','height10m'])
    wind10_month = array['WIND10'].resample(time = '1M').mean().squeeze(['time','height10m'])
    return u10_month, v10_month, wind10_month

u10_month_irri_GAR, v10_month_irri_GAR, wind10_month_irri_GAR = calc_uv10_mean(po_irri_GAR_split)
u10_month_noirri_GAR, v10_month_noirri_GAR, wind10_month_noirri_GAR = calc_uv10_mean(po_noirri_GAR_split)

u10_month_irri_SGAR, v10_month_irri_SGAR, wind10_month_irri_SGAR = calc_uv10_mean(po_irri_SGAR)
u10_month_noirri_SGAR, v10_month_noirri_SGAR, wind10_month_noirri_SGAR = calc_uv10_mean(po_noirri_SGAR)

u10_month_diff_GAR = u10_month_noirri_GAR - u10_month_irri_GAR
v10_month_diff_GAR = v10_month_noirri_GAR - v10_month_irri_GAR

u10_month_diff_SGAR = u10_month_noirri_SGAR - u10_month_irri_SGAR
v10_month_diff_SGAR = v10_month_noirri_SGAR - v10_month_irri_SGAR

wind10_month_diff_GAR = wind10_month_irri_GAR - wind10_month_noirri_GAR
wind10_month_diff_SGAR = wind10_month_irri_SGAR - wind10_month_noirri_SGAR

# PBL mean
pbl_irri_GAR    = po_irri_GAR_split.GHPBL
pbl_noirri_GAR  = po_noirri_GAR_split.GHPBL
pbl_irri_SGAR   = po_irri_SGAR.GHPBL
pbl_noirri_SGAR = po_noirri_SGAR.GHPBL

# convert gpm in m 
def convert_gpm2m(gpm):
    g0 = 9.80665  # Gravity (m/s²)
    Re = 6371000  # Earth radius in meters
    return (Re * gpm) / (Re - gpm)

#conversion necessary
pblh_irri_GAR    = convert_gpm2m(pbl_irri_GAR )
pblh_noirri_GAR  = convert_gpm2m(pbl_noirri_GAR )
pblh_irri_SGAR   = convert_gpm2m(pbl_irri_SGAR )
pblh_noirri_SGAR = convert_gpm2m(pbl_noirri_SGAR )

# create mean value for spatial plot 
pblh_mean_irri_GAR     = pblh_irri_GAR.mean('time')
pblh_mean_noirri_GAR   = pblh_noirri_GAR.mean('time')
pblh_mean_irri_SGAR    = pblh_irri_SGAR.mean('time')
pblh_mean_noirri_SGAR  = pblh_noirri_SGAR.mean('time')

diff_pblh_GAR = pblh_mean_irri_GAR  - pblh_mean_noirri_GAR
diff_pblh_SGAR = pblh_mean_irri_SGAR- pblh_mean_noirri_SGAR

# PS mean
ps_month_mean_irri_SGAR   = po_irri_SGAR['PS'].resample(time = '1M').mean().squeeze(['time'])
ps_month_mean_noirri_SGAR = po_noirri_SGAR['PS'].resample(time = '1M').mean().squeeze(['time'])
ps_month_mean_irri_GAR    = po_irri_GAR_split['PS'].resample(time = '1M').mean().squeeze(['time'])
ps_month_mean_noirri_GAR  = po_noirri_GAR_split['PS'].resample(time = '1M').mean().squeeze(['time'])

diff_ps_GAR  = ps_month_mean_irri_GAR - ps_month_mean_noirri_GAR
diff_ps_SGAR = ps_month_mean_irri_SGAR - ps_month_mean_noirri_SGAR


vec_density = 4
############################################ PLOT FIGURE C1 ##################################
fig = plt.figure(figsize=(15,9))

cax1 = fig.add_axes([0.32, 0.645, 0.37, 0.03])
levels_precip = np.arange(0,220,20)
ticks_precip = np.arange(0,220,20)

cax2 = fig.add_axes([0.32, 0.37, 0.37, 0.03])
levels_wind = np.arange(0,5.5,0.5)
ticks_wind = np.arange(0,6,1)

cax3 = fig.add_axes([0.32, 0.08, 0.37, 0.03])
levels_pblh = np.arange(0,2800,200)
ticks_pblh = np.arange(0,3000,1000)


ax1 = fig.add_subplot(3, 4, 1, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_irri_GAR, ax1, cax1, 'mm', 'precipitation [mm]', 'YlGnBu', levels_precip, ticks_precip, 'max', 'horizontal')
ax1.set_title('')
ax1.text(0.035, 0.964, '0.11° irri', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=14)

ax2 = fig.add_subplot(3, 4, 2, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_noirri_GAR, ax2, cax1, 'mm', 'precipitation [mm]', 'YlGnBu', levels_precip, ticks_precip, 'max', 'horizontal')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.11° noirri', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.0, 1.08, '(b)', transform=ax2.transAxes, fontsize=14)

ax3 = fig.add_subplot(3, 4, 3, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_irri_SGAR, ax3, cax1, 'mm', 'precipitation [mm]', 'YlGnBu', levels_precip, ticks_precip, 'max', 'horizontal')
ax3.set_title('')
ax3.text(0.035, 0.964, '0.0275° irri', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax3.text(0.0, 1.08, '(c)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(3, 4, 4, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_noirri_SGAR, ax4, cax1, 'mm', 'precipitation [mm]', 'YlGnBu', levels_precip, ticks_precip, 'max', 'horizontal')
ax4.set_title('')
ax4.text(0.035, 0.964, '0.0275° noirri', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax4.text(0.0, 1.08, '(d)', transform=ax4.transAxes, fontsize=14)

# windspeed 10 m
ax5 = fig.add_subplot(3, 4, 5, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_irri_GAR, ax5, cax2, 'mm', '10 mwindspeed [ms$^{-1}$]', 'viridis', levels_wind, ticks_wind, 'max', 'horizontal')
ax5.set_title('')
ax5.text(0.035, 0.964, '0.11° irri', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax5.text(0.0, 1.08, '(e)', transform=ax5.transAxes, fontsize=14)
vecplot = ax5.quiver(v10_month_irri_GAR.rlon[::vec_density], u10_month_irri_GAR.rlat[::vec_density], 
                     u10_month_irri_GAR[::vec_density,::vec_density], v10_month_irri_GAR[::vec_density,::vec_density], 
                     scale=45, color="black",  angles='uv')

ax6 = fig.add_subplot(3, 4, 6, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_noirri_GAR, ax6, cax2, 'mm', '10 m windspeed  [ms$^{-1}$]', 'viridis', levels_wind, ticks_wind, 'max', 'horizontal')
ax6.set_title('')
ax6.text(0.035, 0.964, '0.11° noirri', transform=ax6.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax6.text(0.0, 1.08, '(f)', transform=ax6.transAxes, fontsize=14)
vecplot = ax6.quiver(v10_month_noirri_GAR.rlon[::vec_density], u10_month_noirri_GAR.rlat[::vec_density], 
                     u10_month_noirri_GAR[::vec_density,::vec_density], v10_month_noirri_GAR[::vec_density,::vec_density], 
                     scale=45, color="black",  angles='uv')

ax7 = fig.add_subplot(3, 4, 7, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_irri_SGAR, ax7, cax2, 'mm', '10 m windspeed [ms$^{-1}$]', 'viridis', levels_wind, ticks_wind, 'max', 'horizontal')
ax7.set_title('')
ax7.text(0.035, 0.964, '0.0275 irri°', transform=ax7.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax7.text(0.0, 1.08, '(g)', transform=ax7.transAxes, fontsize=14)
vecplot = ax7.quiver(v10_month_irri_SGAR.rlon[::vec_density], u10_month_irri_SGAR.rlat[::vec_density], 
                     u10_month_irri_SGAR[::vec_density,::vec_density], v10_month_irri_SGAR[::vec_density,::vec_density], 
                     scale=45, color="black",  angles='uv')

ax8 = fig.add_subplot(3, 4, 8, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_noirri_SGAR, ax8, cax2, 'mm', '10 m windspeed [ms$^{-1}$]', 'viridis', levels_wind, ticks_wind, 'max', 'horizontal')
ax8.set_title('')
ax8.text(0.035, 0.964, '0.0275 noirri°', transform=ax8.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax8.text(0.0, 1.08, '(h)', transform=ax8.transAxes, fontsize=14)
vecplot = ax8.quiver(v10_month_noirri_SGAR.rlon[::vec_density], u10_month_noirri_SGAR.rlat[::vec_density], 
                     u10_month_noirri_SGAR[::vec_density,::vec_density], v10_month_noirri_SGAR[::vec_density,::vec_density], 
                     scale=45, color="black",  angles='uv')

vref = ax1.quiverkey(vecplot, 0.78, -0.1, 2,
                        r'$2{ms^{-1}}$',
                        labelpos='E',
                        zorder=5, coordinates = 'axes')

ax9 = fig.add_subplot(3, 4, 9, projection=rotated_pole)
rotplot = plot_rotvar(fig, pblh_mean_irri_GAR, ax9, cax3, 'mm', 'PBL [m]', 'plasma', levels_pblh, ticks_pblh, 'max', 'horizontal')
ax9.set_title('')
ax9.text(0.035, 0.964, '0.11 irri°', transform=ax9.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax9.text(0.0, 1.08, '(i)', transform=ax9.transAxes, fontsize=14)

ax10 = fig.add_subplot(3, 4, 10, projection=rotated_pole)
rotplot = plot_rotvar(fig, pblh_mean_noirri_GAR, ax10, cax3, 'mm', 'PBL [m]', 'plasma', levels_pblh, ticks_pblh, 'max', 'horizontal')
ax10.set_title('')
ax10.text(0.035, 0.964, '0.11 noirri°', transform=ax10.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax10.text(0.0, 1.08, '(j)', transform=ax10.transAxes, fontsize=14)

ax11 = fig.add_subplot(3, 4, 11, projection=rotated_pole)
rotplot = plot_rotvar(fig, pblh_mean_irri_SGAR, ax11, cax3, 'mm', 'PBL [m]', 'plasma', levels_pblh, ticks_pblh, 'max', 'horizontal')
ax11.set_title('')
ax11.text(0.035, 0.964, '0.0275 irri°', transform=ax11.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax11.text(0.0, 1.08, '(k)', transform=ax11.transAxes, fontsize=14)

ax12 = fig.add_subplot(3, 4, 12, projection=rotated_pole)
rotplot = plot_rotvar(fig, pblh_mean_noirri_SGAR, ax12, cax3, 'mm', 'PBL [m]', 'plasma', levels_pblh, ticks_pblh, 'max', 'horizontal')
ax12.set_title('')
ax12.text(0.035, 0.964, '0.0275 noirri°', transform=ax12.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax12.text(0.0, 1.08, '(l)', transform=ax12.transAxes, fontsize=14)

fig.subplots_adjust(hspace = 0.5)
plt.savefig(str(dir_out)+'/precip_wind_pblh_abs.png',dpi=300, bbox_inches='tight')



import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


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


################################### PLOT FIGURE 12 ######################################

# differences 
vec_density = 6


fig = plt.figure(figsize=(10,12))
params = {
    "legend.fontsize": 10,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(params)

#cax1 = fig.add_axes([0.315, 0.67, 0.37, 0.02])
cax1 = fig.add_axes([0.915, 0.745, 0.02, 0.14])
levels_precip = list(filter(lambda num: num != 0, np.arange(-50,60,10)))
ticks_precip = np.arange(-40,60,20)

cax2 = fig.add_axes([0.915, 0.422, 0.02, 0.14])
levels_wind = list(filter(lambda num: num != 0, np.arange(-0.8,0.9,0.1)))
ticks_wind = np.arange(-0.8,1.2,0.4).round(1)

cax3 = fig.add_axes([0.915, 0.266, 0.02, 0.14])
levels_pbl = list(filter(lambda num: num != 0, np.arange(-300,350,50)))
ticks_pbl = np.arange(-300,400,100)

cax4 = fig.add_axes([0.915, 0.11, 0.02, 0.14])
levels_ps = list(filter(lambda num: num != 0, np.arange(-50,60,10)))
ticks_ps = np.arange(-40,60,20)


ax1 = fig.add_subplot(5, 2, 1, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_diff_GAR, ax1, cax1, 'mm', 'Δ precipitation [mm]', 'RdBu_r', levels_precip, ticks_precip, 'both', 'vertical')
ax1.set_title('')
ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=14)


ax2 = fig.add_subplot(5, 2, 2, projection=rotated_pole)
rotplot = plot_rotvar(fig, precip_month_diff_SGAR, ax2, cax1, 'mm', 'Δ precipitation [mm]', 'RdBu_r', levels_precip, ticks_precip, 'both', 'vertical')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.0, 1.08, '(b)', transform=ax2.transAxes, fontsize=14)

ax3 = fig.add_subplot(5, 2, 5, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_diff_GAR, ax3, cax2, 'mm', 'Δ windspeed [ms$^{-1}$]', 'RdBu_r', levels_wind, ticks_wind, 'both', 'vertical')
ax3.set_title('')
ax3.text(0.035, 0.964, '0.11°', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax3.text(0.0, 1.08, '(e)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(5, 2, 6, projection=rotated_pole)
rotplot = plot_rotvar(fig, wind10_month_diff_SGAR, ax4, cax2, 'mm', 'Δ windspeed [ms$^{-1}$]', 'RdBu_r', levels_wind, ticks_wind, 'both', 'vertical')
ax4.set_title('')
ax4.text(0.035, 0.964, '0.0275°', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax4.text(0.0, 1.08, '(f)', transform=ax4.transAxes, fontsize=14)

ax5 = fig.add_subplot(5, 2, 3, projection=rotated_pole)
ax5.add_feature(land, zorder=1)
ax5.add_feature(coastline, edgecolor="black", linewidth=0.6)
ax5.add_feature(borders, edgecolor="black", linewidth=0.6)
ax5.set_xmargin(0)
ax5.set_ymargin(0)
ax5.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
ax5.set_title('')
ax5.text(0.035, 0.964, '0.11°', transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax5.text(0.0, 1.08, '(c)', transform=ax5.transAxes, fontsize=14)
vecplot1 = ax5.quiver(v10_month_noirri_GAR.rlon[::vec_density], u10_month_noirri_GAR.rlat[::vec_density], 
                     u10_month_noirri_GAR[::vec_density,::vec_density], v10_month_noirri_GAR[::vec_density,::vec_density], 
                     scale=30, color="red",  angles='uv')

vecplot2 = ax5.quiver(v10_month_irri_GAR.rlon[::vec_density], u10_month_irri_GAR.rlat[::vec_density], 
                     u10_month_irri_GAR[::vec_density,::vec_density], v10_month_irri_GAR[::vec_density,::vec_density], 
                     scale=30, color="blue",  angles='uv')

ax6 = fig.add_subplot(5, 2, 4, projection=rotated_pole)
ax6.set_title('')
ax6.add_feature(land, zorder=1)
ax6.add_feature(coastline, edgecolor="black", linewidth=0.6)
ax6.add_feature(borders, edgecolor="black", linewidth=0.6)
ax6.set_xmargin(0)
ax6.set_ymargin(0)
ax6.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
ax6.text(0.035, 0.964, '0.0275°', transform=ax6.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax6.text(0.0, 1.08, '(d)', transform=ax6.transAxes, fontsize=14)
vecplot1 = ax6.quiver(v10_month_noirri_SGAR.rlon[::vec_density], u10_month_noirri_SGAR.rlat[::vec_density], 
                     u10_month_noirri_SGAR[::vec_density,::vec_density], v10_month_noirri_SGAR[::vec_density,::vec_density], 
                     scale=30, color="red",  angles='uv')

vecplot2 = ax6.quiver(v10_month_irri_SGAR.rlon[::vec_density], u10_month_irri_SGAR.rlat[::vec_density], 
                     u10_month_irri_SGAR[::vec_density,::vec_density], v10_month_irri_SGAR[::vec_density,::vec_density], 
                     scale=30, color="blue",  angles='uv')

vref = ax6.quiverkey(vecplot1, 1.1, 0.3, 2, '', 
                     labelpos='E', zorder=9, coordinates='axes')
ax6.text(1.15, 0.3, "not irrigated\n2 ms$^{-1}$",va='center', ha='left', transform=ax6.transAxes)

vref = ax6.quiverkey(vecplot2, 1.1, 0.1, 2, '', 
                     labelpos='E', zorder=9, coordinates='axes')
#line break didn't work. Therefore, added text manually
ax6.text(1.15, 0.1, "irrigated\n2 ms$^{-1}$", va='center', ha='left', transform=ax6.transAxes)

#PBL
ax9 = fig.add_subplot(5, 2, 9, projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_pblh_GAR, ax9, cax4, 'mm', 'Δ PBL [m]', 'RdBu_r', levels_pbl, ticks_pbl, 'both', 'vertical')
ax9.set_title('')
ax9.text(0.035, 0.964, '0.11°', transform=ax9.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax9.text(0.0, 1.08, '(i)', transform=ax9.transAxes, fontsize=14)

ax10 = fig.add_subplot(5, 2, 10, projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_pblh_SGAR , ax10, cax4, 'mm', 'Δ PBL [m]', 'RdBu_r', levels_pbl, ticks_pbl, 'both', 'vertical')
ax10.set_title('')
ax10.text(0.035, 0.964, '0.0275°', transform=ax10.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax10.text(0.0, 1.08, '(j)', transform=ax10.transAxes, fontsize=14)

#PS
ax7 = fig.add_subplot(5, 2, 7, projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_ps_GAR, ax7, cax3, 'mm', 'Δ PS [Pa]', 'RdBu_r', levels_ps, ticks_ps, 'both', 'vertical')
ax7.set_title('')
ax7.text(0.035, 0.964, '0.11°', transform=ax7.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax7.text(0.0, 1.08, '(g)', transform=ax7.transAxes, fontsize=14)

ax8 = fig.add_subplot(5, 2, 8, projection=rotated_pole)
rotplot = plot_rotvar(fig, diff_ps_SGAR , ax8, cax3, 'mm', 'Δ PS [Pa]', 'RdBu_r', levels_ps, ticks_ps, 'both', 'vertical')
ax8.set_title('')
ax8.text(0.035, 0.964, '0.0275°', transform=ax8.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax8.text(0.0, 1.08, '(h)', transform=ax8.transAxes, fontsize=14)

plt.tight_layout 
plt.savefig(str(dir_out)+'/precip_wind_vectors_transparent_pbl_ps_new.png',dpi=300, bbox_inches='tight')



######################################### PBL and TKE ################################
# # PBL and TKE analysis for Day and Night for explaining of T2Max and T2Min behaviour 

def select_day_night(var, time):
    if time == 'night': 
        hours = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    elif time == 'day': 
         hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

            
    SGAR_irri   = po_irri_SGAR.sel(time=po_irri_SGAR.time.dt.hour.isin(hours))
    SGAR_noirri = po_noirri_SGAR.sel(time=po_noirri_SGAR.time.dt.hour.isin(hours))
    GAR_irri    = po_irri_GAR_split.sel(time=po_irri_GAR_split.time.dt.hour.isin(hours))
    GAR_noirri  = po_noirri_GAR_split.sel(time=po_noirri_GAR_split.time.dt.hour.isin(hours))
    
    var_isobaric_irri_GAR    = GAR_irri[var].mean('time')
    var_isobaric_noirri_GAR  = GAR_noirri[var].mean('time')
    var_isobaric_irri_SGAR   = SGAR_irri[var].mean('time')
    var_isobaric_noirri_SGAR = SGAR_noirri[var].mean('time')   
    return var_isobaric_irri_GAR, var_isobaric_noirri_GAR, var_isobaric_irri_SGAR, var_isobaric_noirri_SGAR

# PBL day vs. night 

# PBL
pbl_irri_GAR    = po_irri_GAR_split.GHPBL
pbl_noirri_GAR  = po_noirri_GAR_split.GHPBL
pbl_irri_SGAR   = po_irri_SGAR.GHPBL
pbl_noirri_SGAR = po_noirri_SGAR.GHPBL

# # convert gpm in m 
def convert_gpm2m(gpm):
    g0 = 9.80665  # Gravity
    Re = 6371000  # Earth radius
    return (Re * gpm) / (Re - gpm)

#conversion necessary
pblh_irri_GAR    = convert_gpm2m(pbl_irri_GAR )
pblh_noirri_GAR  = convert_gpm2m(pbl_noirri_GAR )
pblh_irri_SGAR   = convert_gpm2m(pbl_irri_SGAR )
pblh_noirri_SGAR = convert_gpm2m(pbl_noirri_SGAR )

pbl_day_GAR_irri, pbl_day_GAR_noirri, pbl_day_SGAR_irri, pbl_day_SGAR_noirri = select_day_night('GHPBL','day')
pbl_night_GAR_irri, pbl_night_GAR_noirri, pbl_night_SGAR_irri, pbl_night_SGAR_noirri = select_day_night('GHPBL','night')

diff_pbl_GAR_day  = pbl_day_GAR_irri  - pbl_day_GAR_noirri
diff_pbl_SGAR_day  = pbl_day_SGAR_irri - pbl_day_SGAR_noirri

diff_pbl_GAR_night  = pbl_night_GAR_irri  - pbl_night_GAR_noirri
diff_pbl_SGAR_night = pbl_night_SGAR_irri - pbl_night_SGAR_noirri

diffdiff_pbl_day   = diff_pbl_SGAR_day   - diff_pbl_GAR_day
diffdiff_pbl_night = diff_pbl_SGAR_night - diff_pbl_GAR_night

# TKE
tke_irri_GAR    = po_irri_GAR_split.TKE
tke_noirri_GAR  = po_noirri_GAR_split.TKE
tke_irri_SGAR   = po_irri_SGAR.TKE
tke_noirri_SGAR = po_noirri_SGAR.TKE

tke_day_GAR_irri, tke_day_GAR_noirri, tke_day_SGAR_irri, tke_day_SGAR_noirri         = select_day_night('TKE','day')
tke_night_GAR_irri, tke_night_GAR_noirri, tke_night_SGAR_irri, tke_night_SGAR_noirri = select_day_night('TKE','night')
lev = 48
diff_tke_GAR_day   = tke_day_GAR_irri[lev]  - tke_day_GAR_noirri[lev]
diff_tke_SGAR_day  = tke_day_SGAR_irri[lev] - tke_day_SGAR_noirri[lev]

diff_tke_GAR_night  = tke_night_GAR_irri[lev]  - tke_night_GAR_noirri[lev]
diff_tke_SGAR_night = tke_night_SGAR_irri[lev] - tke_night_SGAR_noirri[lev]

diffdiff_tke_day   = diff_tke_SGAR_day   - diff_tke_GAR_day
diffdiff_tke_night = diff_tke_SGAR_night - diff_tke_GAR_night

############################ PLOT FIGURE 11 #############################
#### NEW #####


# plot as subplot 
# diff 

plot_diff_11_list   = [diff_tke_GAR_day,  diff_tke_GAR_night]
plot_diff_0275_list = [diff_tke_SGAR_day, diff_tke_SGAR_night]
plot_diffdiff_list  = [diffdiff_tke_day, diffdiff_tke_night]
title_list          = ['TKE, day', 'TKE, night']



fig_name_11    = ['(a)', '(d)']
fig_name_0275  = ['(b)', '(e)']
fig_name_diff  = ['(c)', '(f)']

params = {
    "legend.fontsize": 10,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(params)

fig = plt.figure(figsize=(18,7))
#fig.suptitle('Irrigation effects in June 2017')
spec = fig.add_gridspec(ncols=3, nrows=2)

#cax1 = fig.add_axes([0.22, 0.55, 0.35, 0.02])
#cax2 = fig.add_axes([0.678, 0.55, 0.22, 0.02])
cax3 = fig.add_axes([0.22, 0.05, 0.35, 0.02])
cax4 = fig.add_axes([0.678, 0.05, 0.22, 0.02])
cax_list = [cax1, cax3]

cax_diff_list = [cax2, cax4]

levels_tke = list(filter(lambda num: num != 0,  np.arange(-0.5,0.55,0.05)))
ticks_tke =  np.arange(-0.5,0.6, 0.1).round(2)

levels_diff_tke = list(filter(lambda num: num != 0,  np.arange(-0.3,0.35,0.05).round(2)))
ticks_diff_tke  =  np.arange(-0.3,0.4, 0.1).round(2)

levels_list = [levels_tke, levels_tke]
ticks_list = [ticks_tke, ticks_tke]

levels_diff_list = [levels_diff_tke, levels_diff_tke]
ticks_diff_list = [ticks_diff_tke, ticks_diff_tke]
label_list = ['Δ TKE [Jkg$^{-1}$]', 'Δ TKE [Jkg$^{-1}$]']

for i, (plot_diff_11, plot_diff_0275, plot_diffdiff, title, label, levels, levels_diff, ticks, ticks_diff, cax, cax_diff) in \
    enumerate(zip(plot_diff_11_list, plot_diff_0275_list, plot_diffdiff_list, title_list, label_list, levels_list, \
                  levels_diff_list, ticks_list, ticks_diff_list, cax_list, cax_diff_list)): 

    # 12 km 
    ax1 = fig.add_subplot(spec[0+i, 0], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar(fig, plot_diff_11, ax1, cax, '°C', label, 'RdBu_r', levels, ticks, 'both', False, 'horizontal')
    ax1.set_title(title)
    ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax1.text(0.0, 1.05, fig_name_11[i], transform=ax1.transAxes, fontsize=14)
    
    # 3 km 
    ax2 = fig.add_subplot(spec[0+i, 1], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar(fig, plot_diff_0275, ax2, cax, '°C', label, 'RdBu_r', levels, ticks, 'both', False, 'horizontal')
    ax2.set_title(title)
    ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax2.text(0.0, 1.05, fig_name_0275[i], transform=ax2.transAxes, fontsize=14)

    # diff 12 - 3 km
    ax3 = fig.add_subplot(spec[0+i, 2], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar(fig, plot_diffdiff, ax3, cax_diff, '°C', label, 'PiYG_r', levels_diff, ticks_diff, 'both', False, 'horizontal')
    ax3.set_title('*diff = 0.0275° - 0.11° (splitted)')
    ax3.text(0.035, 0.964, 'diff*', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax3.text(0.0, 1.05, fig_name_diff[i], transform=ax3.transAxes, fontsize=14)
    # Overlay mask on the existing plot
    #hatchplot = ax3.contourf(irrifrac_change['rlon'], irrifrac_change['rlat'], irrifrac_change, levels=[-0.5, 0.5, 1.5],hatches=['....', '/////'], colors='none', alpha=0, 
    #            zorder = 7)
        
plt.subplots_adjust(hspace = 0.7)
plt.savefig(dir_out+'/tke_diff_day_night_spatial_new.png', dpi=300, bbox_inches='tight')

################################### PLOT B1 ##############################################################
# absolut values 
fig = plt.figure(figsize=(15,12))

cax1 = fig.add_axes([0.33, 0.51, 0.5, 0.02])
levels_pbl = np.arange(0,3200,200)
ticks_pbl = np.arange(0,4000,1000)

cax2 = fig.add_axes([0.33, 0.1, 0.5, 0.02])
levels_tke = np.arange(0,1.1,0.1)
ticks_tke = np.arange(0,1.5,0.5)


cmap = 'plasma'

# PBL day
ax1 = fig.add_subplot(4, 4, 1, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_day_GAR_irri, ax1, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax1.set_title('')
ax1.text(0.035, 0.964, '0.11° irri day', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=14)

ax2 = fig.add_subplot(4, 4, 2, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_day_GAR_noirri, ax2, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.11° noirri day', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax2.text(0.0, 1.08, '(b)', transform=ax2.transAxes, fontsize=14)

ax3 = fig.add_subplot(4, 4, 3, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_day_SGAR_irri, ax3, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax3.set_title('')
ax3.text(0.035, 0.964, '0.0275° irri day', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax3.text(0.0, 1.08, '(c)', transform=ax3.transAxes, fontsize=14)

ax4 = fig.add_subplot(4, 4, 4, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_day_SGAR_noirri, ax4, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax4.set_title('')
ax4.text(0.035, 0.964, '0.0275° noirri day', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax4.text(0.0, 1.08, '(d)', transform=ax4.transAxes, fontsize=14)

# PBL night
ax5 = fig.add_subplot(4, 4, 5, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_night_GAR_irri, ax5, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax5.set_title('')
ax5.text(0.035, 0.964, '0.11° irri night', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax5.text(0.0, 1.08, '(e)', transform=ax5.transAxes, fontsize=14)


ax6 = fig.add_subplot(4, 4, 6, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_night_GAR_noirri, ax6, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax6.set_title('')
ax6.text(0.035, 0.964, '0.11° noirri night', transform=ax6.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax6.text(0.0, 1.08, '(f)', transform=ax6.transAxes, fontsize=14)

ax7 = fig.add_subplot(4, 4, 7, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_night_SGAR_irri, ax7, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax7.set_title('')
ax7.text(0.035, 0.964, '0.0275 irri night', transform=ax7.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax7.text(0.0, 1.08, '(g)', transform=ax7.transAxes, fontsize=14)


ax8 = fig.add_subplot(4, 4, 8, projection=rotated_pole)
rotplot = plot_rotvar(fig, pbl_night_SGAR_noirri, ax8, cax1, 'mm', 'PBL [m]', cmap, levels_pbl, ticks_pbl, 'max', 'horizontal')
ax8.set_title('')
ax8.text(0.035, 0.964, '0.0275 noirri night', transform=ax8.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax8.text(0.0, 1.08, '(h)', transform=ax8.transAxes, fontsize=14)

###########################

# TKE day
ax9 = fig.add_subplot(4, 4, 9, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_day_GAR_irri[48], ax9, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax9.set_title('')
ax9.text(0.035, 0.964, '0.11° irri day', transform=ax9.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax9.text(0.0, 1.08, '(i)', transform=ax9.transAxes, fontsize=14)

ax10 = fig.add_subplot(4, 4, 10, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_day_GAR_noirri[48], ax10, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax10.set_title('')
ax10.text(0.035, 0.964, '0.11° noirri day', transform=ax10.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax10.text(0.0, 1.08, '(j)', transform=ax10.transAxes, fontsize=14)

ax11 = fig.add_subplot(4, 4, 11, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_day_SGAR_irri[48], ax11, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax11.set_title('')
ax11.text(0.035, 0.964, '0.0275° irri day', transform=ax11.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax11.text(0.0, 1.08, '(k)', transform=ax11.transAxes, fontsize=14)

ax12 = fig.add_subplot(4, 4, 12, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_day_SGAR_noirri[48], ax12, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax12.set_title('')
ax12.text(0.035, 0.964, '0.0275° noirri day', transform=ax12.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax12.text(0.0, 1.08, '(l)', transform=ax12.transAxes, fontsize=14)

# TKE night
ax13 = fig.add_subplot(4, 4,13, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_night_GAR_irri[48], ax13, cax2, 'mm','TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax13.set_title('')
ax13.text(0.035, 0.964, '0.11° irri night', transform=ax13.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax13.text(0.0, 1.08, '(m)', transform=ax13.transAxes, fontsize=14)

ax14 = fig.add_subplot(4, 4, 14, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_night_GAR_noirri[48], ax14, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax14.set_title('')
ax14.text(0.035, 0.964, '0.11° noirri night', transform=ax14.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax14.text(0.0, 1.08, '(n)', transform=ax14.transAxes, fontsize=14)

ax15 = fig.add_subplot(4, 4, 15, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_night_SGAR_irri[48], ax15, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax15.set_title('')
ax15.text(0.035, 0.964, '0.0275 irri° night', transform=ax15.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax15.text(0.0, 1.08, '(o)', transform=ax15.transAxes, fontsize=14)

ax16 = fig.add_subplot(4, 4, 16, projection=rotated_pole)
rotplot = plot_rotvar(fig, tke_night_SGAR_noirri[48], ax16, cax2, 'mm', 'TKE [Jkg$^{-1}$]', cmap, levels_tke, ticks_tke, 'max', 'horizontal')
ax16.set_title('')
ax16.text(0.035, 0.964, '0.0275 noirri° night', transform=ax16.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax16.text(0.0, 1.08, '(p)', transform=ax16.transAxes, fontsize=14)

#fig.subplots_adjust(hspace = 0.3)
plt.savefig(str(dir_out)+'/pbl_tke_abs.png',dpi=300, bbox_inches='tight')





