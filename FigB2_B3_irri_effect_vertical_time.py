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

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
cax = fig.add_axes([1.05, 0.12, 0.02, 0.8])

#GAR 
levels = list(filter(lambda num: num != 0, np.arange(-2, 2.2, 0.2)))
plot1 = diff_GAR_day.__xarray_dataarray_variable__.T.plot(ax=ax1, levels=levels, add_colorbar = False)
ax1.invert_yaxis()
ax1.set_xlabel("Hour")
ax1.set_ylim(100500, 70000)
ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.set_title('')
#SGAR
plot2 = diff_SGAR_day.__xarray_dataarray_variable__.T.plot(ax=ax2, levels=levels, add_colorbar = False )
ax2.invert_yaxis()
ax2.set_xlabel("Hour")
ax2.set_ylim(100500, 70000)
ax2.set_ylabel('')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
cbar = fig.colorbar(
           plot2, cax=cax, label='Δ temperature [K]')
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig(dir_out+'/vertical_cooling_hours_appendix.png', dpi=300, bbox_inches='tight')

# specific humidity 
diff_GAR  = po_irri_GAR_splitt.QD - po_noirri_GAR_splitt.QD
diff_SGAR = po_irri_SGAR.QD - po_noirri_SGAR.QD
diff_GAR_day  = ((diff_GAR.groupby('time.hour').mean('time').where((mask_night>0.5) | (mask_day>0.5))).mean(['rlat','rlon']))*1000
diff_SGAR_day = ((diff_SGAR.groupby('time.hour').mean('time').where((mask_night>0.5)| (mask_day>0.5))).mean(['rlat','rlon']))*1000

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

cax = fig.add_axes([1.05, 0.12, 0.02, 0.8])

#GAR 
levels = list(filter(lambda num: num != 0, np.arange(-2.0, 2.2, 0.2)))
plot1 = diff_GAR_day.__xarray_dataarray_variable__.T.plot(ax=ax1, levels=levels, extend = 'both', add_colorbar = False)
ax1.invert_yaxis()
ax1.set_xlabel("Hour")
ax1.set_ylim(100500, 70000)
ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
ax1.set_title('')

#SGAR
plot2 = diff_SGAR_day.__xarray_dataarray_variable__.T.plot(ax=ax2, levels=levels, extend = 'both', add_colorbar = False )
ax2.invert_yaxis()
ax2.set_xlabel("Hour")
ax2.set_ylim(100500, 70000)
ax2.set_ylabel('')
ax2.set_title('')
ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
cbar = fig.colorbar(
           plot2, cax=cax, label='Δ specific humdity [g/kg]', extend = 'both')
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig(dir_out+'/vertical_moisture_hours_appendix.png', dpi=300, bbox_inches='tight')




