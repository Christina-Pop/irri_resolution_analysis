#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd

print('working directory',os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

from functions_correcting_time import correct_timedim, correct_timedim_mfiles
from functions_plotting import plot_rotvar, plot_rotvar_adjust_cbar, rotated_pole
from functions_reading_files import read_efiles, read_mfiles, read_efiles_in_chunks 

# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

# read data
year = 2017
month = 0

#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"

varlist_SGAR = ["APRL", "ALAI_PFI"]
var_num_list_SGAR = [ "142", "746"]

varlist_GAR = ["APRL","APRC" ,"ALAI_PFI"]
var_num_list_GAR = [ "142","143","746"]

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

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR])

ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR])

dsirr_newtime_SGAR       = correct_timedim(ds_var_irri_SGAR)
dsnoirr_newtime_SGAR     = correct_timedim(ds_var_noirri_SGAR)

dsirr_newtime_GAR = correct_timedim(ds_var_irri_GAR)
dsnoirr_newtime_GAR = correct_timedim(ds_var_noirri_GAR)

# use function cut_same_area 

def cut_same_area(source_area, target_area):
    min_rlon_target=target_area.rlon[0].values 
    max_rlon_target=target_area.rlon[-1].values

    min_rlat_target=target_area.rlat[0].values 
    max_rlat_target=target_area.rlat[-1].values
    
    cutted_ds = source_area.sel(rlon=slice(min_rlon_target,max_rlon_target), \
                                rlat=slice(min_rlat_target,max_rlat_target))
    return cutted_ds

 # select months
dsnoirr_newtime_GAR = dsnoirr_newtime_GAR.sel(time=dsnoirr_newtime_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_GAR   = dsirr_newtime_GAR.sel(time=dsirr_newtime_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_SGAR      = dsirr_newtime_SGAR.sel(time=dsirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsnoirr_newtime_SGAR    = dsnoirr_newtime_SGAR .sel(time=dsnoirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))


# cut out the Po valley 
po_irri_GAR    = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_GAR  = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_SGAR = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR)
po_irri_SGAR   = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR)

# split GAR to be able to use the filter 
po_noirri_GAR_split = po_noirri_GAR.interp_like(
    po_noirri_SGAR,
    method='nearest')
po_irri_GAR_split = po_irri_GAR.interp_like(
    po_irri_SGAR,
    method='nearest')

# check precipitation 
#select grid cells
irrilimit = 0.70
lai_limit = 0.1

dsirr_newtime_GAR_cut_filter = po_irri_GAR_split.where((po_irri_SGAR.IRRIFRAC>irrilimit) & (po_irri_GAR_split.ALAI_PFI > lai_limit ))
dsnoirr_newtime_GAR_cut_filter = po_noirri_GAR_split.where((po_noirri_SGAR.IRRIFRAC>irrilimit) & (po_noirri_GAR_split.ALAI_PFI > lai_limit ))

po_irri_SGAR_filter = po_irri_SGAR.where((po_irri_SGAR.IRRIFRAC>irrilimit) & (po_irri_SGAR.ALAI_PFI > lai_limit ))
po_noirri_SGAR_filter = po_noirri_SGAR.where((po_noirri_SGAR.IRRIFRAC>irrilimit) & (po_noirri_SGAR.ALAI_PFI > lai_limit ))

precip_SGAR_irri = po_irri_SGAR_filter.APRL.resample(time = 'M').sum().sum(['rlon','rlat'])
precip_SGAR_irri = precip_SGAR_irri.squeeze('pft_irr')
precip_SGAR_noirri = po_noirri_SGAR_filter.APRL.resample(time = 'M').sum().sum(['rlon','rlat'])
precip_SGAR_noirri = precip_SGAR_noirri.squeeze('pft_irr')

precip_GAR_irri = (dsirr_newtime_GAR_cut_filter.APRL+dsirr_newtime_GAR_cut_filter.APRC).resample(time = 'M').sum().sum(['rlon','rlat'])
precip_GAR_irri = precip_GAR_irri.squeeze('pft_irr')
precip_GAR_noirri = (dsnoirr_newtime_GAR_cut_filter.APRL+dsnoirr_newtime_GAR_cut_filter.APRC).resample(time = 'M').sum().sum(['rlon','rlat'])
precip_GAR_noirri = precip_GAR_noirri.squeeze('pft_irr')

df = pd.DataFrame({
    "GAR_irri": precip_GAR_irri.to_pandas().values,
    "GAR_noirri": precip_GAR_noirri.to_pandas().values,
    "SGAR_irri": precip_SGAR_irri.to_pandas().values,
    "SGAR_noirri": precip_SGAR_noirri.to_pandas().values,
}, index=precip_SGAR_irri.time.to_pandas())
df = df.head(5)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)  

ax = fig.add_subplot(gs[0]) 
legend_ax = fig.add_subplot(gs[1]) 
legend_ax.axis('off')  

# --- Plot bar chart ---
df.plot(kind="bar", width=0.8, ax=ax, legend=False)

# add hatches and patches 
colors = ['cornflowerblue', 'darkorange', 'white', 'white']
hatches = [None, None, '///', '///']
hatchcolors = [None, None, 'cornflowerblue', 'darkorange']

iter_colors = np.repeat(colors, 5)
iter_hatches = np.repeat(hatches, 5)
iter_hatchcolors = np.repeat(hatchcolors, 5)

for patch, color, hatch, hatchcolor in zip(ax.patches, iter_colors, iter_hatches, iter_hatchcolors):
    patch.set_facecolor(color)
    patch.set_hatch(hatch)
    patch.set_edgecolor(hatchcolor)

# Legends
# Simulation
sim_labels = ['irrigated', 'not irrigated']
sim_handles = [Patch(facecolor='cornflowerblue'),
               Patch(facecolor='darkorange')]
# Resolution 
res_labels = ['0.11°', '0.0275°']
res_handles = [Patch(facecolor='white', edgecolor='black'),
               Patch(facecolor='white', edgecolor='black', hatch='///')]

sim_legend = ax.legend(sim_handles, sim_labels, title='simulation',
                       loc='upper left', bbox_to_anchor=(1.00, 1.0),
                       frameon=True)

ax.add_artist(sim_legend)  

res_legend = ax.legend(res_handles, res_labels, title='resolution',
                       loc='upper left', bbox_to_anchor=(1.00, 0.8),
                       frameon=True, labelspacing=0.65)

for patch in sim_legend.get_patches():
    patch.set_height(8)
    patch.set_y(-3)

ax.set_xlabel("Month")
ax.set_ylabel("Precipitation sum in region [mm month$^{-1}$]")
ax.set_xticks(range(len(df.index)))
ax.set_xticklabels(df.index.strftime("%b-%Y"), rotation=45)

plt.savefig(dir_out+'/FigC1_precipitation_sum_monthly.png', dpi=300)




