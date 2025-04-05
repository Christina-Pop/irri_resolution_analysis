#!/usr/bin/env python
# coding: utf-8
# Cretes Figure 4 for the diurnal surface energy balance

import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns

from functions_correcting_time import correct_timedim
from functions_plotting import rotated_pole
from functions_reading_files import read_efiles, read_efiles_in_chunks 
print('working directory',os.getcwd())

# create plotting directory 
dir_working = os.getcwd()
# creates dir in parent directory
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)


####################### READ DATA ######################################

# read data
year = 2017
month = 0
#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"
#
varlist_SGAR = [ "ALAI_PFI" , "AHFSIRR", "AHFLIRR", "SRADS", "TRADS"]
var_num_list_SGAR = [ "746", "734", "736", "176", "177"]
#
varlist_GAR = [ "ALAI_PFI", "AHFSIRR", "AHFLIRR", "SRADS", "TRADS"]
var_num_list_GAR = [ "746", "734", "736", "176", "177"]


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

# adding irrifrac 
irrifrac_file_SGAR = str(data_path_SGAR_noirri)+'/'+str(exp_number_noirri_SGAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_SGAR)+'e_c743_201706.nc'
irrifrac_data_SGAR = xr.open_dataset(irrifrac_file_SGAR)
irrifrac_SGAR = irrifrac_data_SGAR.IRRIFRAC[0]

irrifrac_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'//var_series/IRRIFRAC/e'+str(exp_number_noirri_GAR)+'e_c743_201706.nc'
irrifrac_data_GAR = xr.open_dataset(irrifrac_file_GAR)
irrifrac_GAR = irrifrac_data_GAR.IRRIFRAC[0]

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR])

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

 # select months
dsnoirr_newtime_GAR = dsnoirr_newtime_GAR.sel(time=dsnoirr_newtime_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_GAR   = dsirr_newtime_GAR.sel(time=dsirr_newtime_GAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsirr_newtime_SGAR     = dsirr_newtime_SGAR.sel(time=dsirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))
dsnoirr_newtime_SGAR   = dsnoirr_newtime_SGAR.sel(time=dsnoirr_newtime_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))

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
#
dsirr_newtime_SGAR   = po_irri_SGAR
dsnoirr_newtime_SGAR = po_noirri_SGAR
#
dsirr_newtime_GAR_split   = po_irri_GAR_split
dsnoirr_newtime_GAR_split = po_noirri_GAR_split


##################### SELECT GRIDCELLS ##############################################

# surface energy balance for one day
# filter out grid cells with irrigated fraction (only relevant for TRADS, SRADS) and LAI 
irrilimit = 0.7
lai_limit = 0.1

dsirr_newtime_SGAR_filter = dsirr_newtime_SGAR.where((dsirr_newtime_SGAR.IRRIFRAC>irrilimit) & (dsirr_newtime_SGAR.ALAI_PFI > lai_limit ))
dsnoirr_newtime_SGAR_filter = dsnoirr_newtime_SGAR.where((dsnoirr_newtime_SGAR.IRRIFRAC>irrilimit) & (dsnoirr_newtime_SGAR.ALAI_PFI > lai_limit ))


dsirr_newtime_GAR_filter = dsirr_newtime_GAR_split.where((dsirr_newtime_SGAR.IRRIFRAC>irrilimit) & (dsirr_newtime_GAR_split.ALAI_PFI > lai_limit ))
dsnoirr_newtime_GAR_filter = dsnoirr_newtime_GAR_split.where((dsnoirr_newtime_SGAR.IRRIFRAC>irrilimit) & (dsnoirr_newtime_GAR_split.ALAI_PFI > lai_limit ))
#########################################################################################################################

####### COMBINE TO DIURNAL CYCLE AND PLOT IN LOOP ###########

ds_GAR_list = [ dsirr_newtime_GAR_filter, dsnoirr_newtime_GAR_filter]
ds_SGAR_list = [dsirr_newtime_SGAR_filter, dsnoirr_newtime_SGAR_filter ]

savetitel_list = ['irrigated', 'not irrigated']
fig_name_list = ['(a)','(b)']
params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

fig, axs = plt.subplots(1, 3,  figsize=(18,4), gridspec_kw={'wspace': 0.3})
plt.rcParams.update(params)

for i, (ds_GAR, ds_SGAR, savetitel, fig_name) in enumerate(zip(ds_GAR_list, ds_SGAR_list, savetitel_list, fig_name_list)): 
    
    # mean diurnal surface energy balance 
    sensible_GAR = -ds_GAR["AHFSIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    latent_GAR   = -ds_GAR["AHFLIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    lw_rad_GAR   = ds_GAR["TRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    sw_rad_GAR   = ds_GAR["SRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    rn_GAR       = lw_rad_GAR + sw_rad_GAR
    ground_GAR   = -sensible_GAR-latent_GAR+(lw_rad_GAR+sw_rad_GAR)

    sensible_SGAR = -ds_SGAR["AHFSIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    latent_SGAR   = -ds_SGAR["AHFLIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    lw_rad_SGAR   = ds_SGAR["TRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    sw_rad_SGAR   = ds_SGAR["SRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    rn_SGAR       = lw_rad_SGAR + sw_rad_SGAR
    ground_SGAR   = -sensible_SGAR-latent_SGAR+(lw_rad_SGAR+sw_rad_SGAR)
    
    
    # plot with seaborn 
    a_1 = sensible_GAR.values
    a_2 = latent_GAR.values
    a_3 = rn_GAR.values 
    a_4 = ground_GAR.values

    b_1 = sensible_SGAR.values
    b_2 = latent_SGAR.values
    b_3 = rn_SGAR.values 
    b_4 = ground_SGAR.values

    data = {
        "date": np.concatenate(
            (
               sensible_GAR.hour.values,
               latent_GAR.hour.values,
               rn_GAR.hour.values,
               ground_GAR.hour.values,
               sensible_SGAR.hour.values,
               latent_SGAR.hour.values,
               rn_SGAR.hour.values,
               ground_SGAR.hour.values,
            )
        ),
        "values": np.concatenate((a_1, a_2, a_3, a_4,
                                  b_1, b_2, b_3, b_4)),
        "variable": ['Qh'] * len(a_1)
        + ['Qe'] * len(a_2) 
        + ['Rn'] * len(a_3)
        + ['G'] * len(a_4)
        + ['Qh'] * len(b_1)
        + ['Qe'] * len(b_2) 
        + ['Rn']*len(b_3)
        + ['G'] * len(b_4),
         "resolution": ['0.11°'] * len(a_1)
        + ['0.11°'] * len(a_2)
        + ['0.11°'] * len(a_3)
        + ['0.11°'] * len(a_4)
        + ['0.0275°'] * len(b_1)
        + ['0.0275°'] * len(b_2)
        + ['0.0275°'] * len(b_3)
        + ['0.0275°'] * len(b_4),}
    
    colors = {'Qh': "green",
              'Qe': "blue",
              'Rn' : "orange",
              'G'  : "brown"}      
    
    p = sns.lineplot(
        data=data,
        x="date",
        y="values",
        hue="variable",
        style="resolution",
        palette=colors,
        ax=axs[i],
        linewidth = 1.0,
    )
    p.legend_.remove()
    p.set_xticklabels(axs[i].get_xticklabels(),rotation=45)
    axs[i].grid(True)
    axs[i].set_ylim(-100, 600)
    axs[i].set_ylabel("Wm$^{-2}$")    
    axs[i].set_xlabel("hour")
    axs[i].text(0.025, 0.964, savetitel, transform=axs[i].transAxes, fontsize=12, va='top', bbox=dict(facecolor='white', edgecolor='black', pad=5.0), zorder=+6)
    axs[i].text(0.0, 1.045, fig_name, transform=axs[i].transAxes, fontsize=14)
    if i == 1:
        p.legend(bbox_to_anchor=(2.7, 1.0))

        
# diff plot
ds_irr_list = [dsirr_newtime_GAR_filter, dsirr_newtime_SGAR_filter]
ds_noirr_list = [dsnoirr_newtime_GAR_filter, dsnoirr_newtime_SGAR_filter]

savetitel_list = ['0.11°', '0.0275°']
style_list = ['solid', 'dashed']

params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


for i, (ds_irr, ds_noirr, savetitel, style) in enumerate(zip(ds_irr_list, ds_noirr_list, savetitel_list, style_list)): 
    
    sensible_irr = -ds_irr["AHFSIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    latent_irr   = -ds_irr["AHFLIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    lw_rad_irr   = ds_irr["TRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    sw_rad_irr   = ds_irr["SRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    rn_irr       = lw_rad_irr + sw_rad_irr
    ground_irr   = -sensible_irr-latent_irr+(lw_rad_irr+sw_rad_irr)

    sensible_noirr = -ds_noirr["AHFSIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    latent_noirr   = -ds_noirr["AHFLIRR"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    lw_rad_noirr   = ds_noirr["TRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    sw_rad_noirr   = ds_noirr["SRADS"].groupby('time.hour').mean(['time','rlat','rlon']).squeeze('pft_irr')
    rn_noirr       = lw_rad_noirr + sw_rad_noirr
    ground_noirr   = -sensible_noirr-latent_noirr+(lw_rad_noirr+sw_rad_noirr)
    
    # mean diurnal surface energy balance 
    sensible_diff  = sensible_irr - sensible_noirr
    latent_diff    = latent_irr   - latent_noirr 
    rn_diff        = rn_irr       - rn_noirr
    ground_diff    = ground_irr   - ground_noirr
    # plot with seaborn 

    # diff plot
    
    a_1 = sensible_diff.values
    a_2 = latent_diff.values
    a_3 = rn_diff.values 
    a_4 = ground_diff.values
    
    data = {
        "date": np.concatenate(
            (
                sensible_diff.hour.values,
                latent_diff.hour.values, 
                rn_diff.hour.values,
                ground_diff.hour.values
            )
        ),
        "values": np.concatenate((a_1, a_2, a_3, a_4)),
         "variable": ['Qh'] * len(a_1)
        + ['Qe'] * len(a_2) 
        + ['Rn'] * len(a_3)
        + ['G'] * len(a_4),
        "resolution": [savetitel] * len(a_1)
        + [savetitel] * len(a_2)
        + [savetitel] * len(a_3)
        + [savetitel] * len(a_4)
       }
    plt.rcParams.update(params)
    colors = {'Qh': "green",
              'Qe': "blue",
              'Rn' : "orange",
              'G'  : "brown"}    

    p = sns.lineplot(
        data=data,
        x="date",
        y="values",
        hue="variable",
        #style="resolution",
        linestyle = style,
        palette=colors,
        ax=axs[2],
        linewidth = 1.0
    )
    p.set_xticklabels(axs[2].get_xticklabels(),rotation=45)
    axs[2].grid(True)
    axs[2].set_ylim(-300, 300)
    axs[2].set_ylabel("Δ Wm$^{-2}$")
    #axs[2].set_xlabel("hour")
    axs[2].get_legend().remove()
    axs[2].text(0.0, 1.045, '(c)', transform=axs[2].transAxes, fontsize=14)
    
    plt.rcParams.update(params)
    plt.tight_layout
    plt.savefig(str(dir_out)+'/surface_energy_balance_diurnal_cycle_combined.png',dpi=300, bbox_inches='tight')






