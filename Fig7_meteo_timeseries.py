#!/usr/bin/env python
# coding: utf-8

# Creates Figure 7: Timeseries for meteorological variables


import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns
import scipy.stats as stats
import matplotlib.dates as mdates
from datetime import datetime

from functions_correcting_time import correct_timedim
from functions_plotting import rotated_pole
from functions_reading_files import read_efiles, read_efiles_in_chunks 
print('working directory',os.getcwd())

# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

############## READ DATA #####################
# read data
year = 2017
month = 0

exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"

varlist_SGAR = [ "TEMP2", "EVAPIRR","ALAI_PFI", "DEW2"]
var_num_list_SGAR = [ "167", "739", "746", "168"]

varlist_GAR = [ "TEMP2", "EVAPIRR","ALAI_PFI", "DEW2"]
var_num_list_GAR = [ "167", "739", "746", "168"]

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

irrifrac_file_GAR = str(data_path_GAR_noirri)+'/'+str(exp_number_noirri_GAR)+'/var_series/IRRIFRAC/e'+str(exp_number_noirri_GAR)+'e_c743_201706.nc'
irrifrac_data_GAR = xr.open_dataset(irrifrac_file_GAR)
irrifrac_GAR = irrifrac_data_GAR.IRRIFRAC[0]

ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR])

dsirr_newtime_SGAR = correct_timedim(ds_var_irri_SGAR)
po_noirri_SGAR = correct_timedim(ds_var_noirri_SGAR)
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
po_noirri_SGAR   = po_noirri_SGAR.sel(time=po_noirri_SGAR.time.dt.month.isin([3, 4, 5, 6, 7]))

# cut out the Po valley 
po_irri_GAR    = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_GAR  = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_SGAR = cut_same_area(po_noirri_SGAR, po_noirri_GAR)
po_irri_SGAR   = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR)

# split GAR to be able to use the filter 
po_noirri_GAR_split = po_noirri_GAR.interp_like(
    po_noirri_SGAR,
    method='nearest')
po_irri_GAR_split = po_irri_GAR.interp_like(
    po_irri_SGAR,
    method='nearest')

###################### CALCULATE RELATIVE HUMIDITY ####################################

# calculate rh 
def calculate_rh(tdew, t):
    # from 
    tdew_cel = tdew -273.15
    t_cel = t-273.15
    rh = 100 * (
        (np.exp((17.625 * tdew_cel) / (243.04 + tdew_cel)))
        / (np.exp((17.625 * t_cel) / (243.04 + t_cel)))
    )
    return rh

po_noirri_SGAR['RH'] = calculate_rh(po_noirri_SGAR.DEW2, po_noirri_SGAR.TEMP2)
po_irri_SGAR['RH']   = calculate_rh(po_irri_SGAR.DEW2, po_irri_SGAR.TEMP2)
po_noirri_GAR_split['RH'] = calculate_rh(po_noirri_GAR_split.DEW2, po_noirri_GAR_split.TEMP2)
po_irri_GAR_split['RH']   = calculate_rh(po_irri_GAR_split.DEW2, po_irri_GAR_split.TEMP2)

################## SELECT RELEVANTE GRIDCELLS #################################

irrilimit = 0.70
lai_limit = 0.1

dsirr_newtime_GAR_cut_filter = po_irri_GAR_split.where((po_irri_SGAR.IRRIFRAC>irrilimit) & (po_irri_GAR_split.ALAI_PFI > lai_limit ))
dsnoirr_newtime_GAR_cut_filter = po_noirri_GAR_split.where((po_noirri_SGAR.IRRIFRAC>irrilimit) & (po_noirri_GAR_split.ALAI_PFI > lai_limit ))

po_irri_SGAR_filter = po_irri_SGAR.where((po_irri_SGAR.IRRIFRAC>irrilimit) & (po_irri_SGAR.ALAI_PFI > lai_limit ))
po_noirri_SGAR_filter = po_noirri_SGAR.where((po_noirri_SGAR.IRRIFRAC>irrilimit) & (po_noirri_SGAR.ALAI_PFI > lai_limit ))


dsirr_newtime_GAR_cut_filter_lai = po_irri_GAR_split.where((po_irri_SGAR.IRRIFRAC>irrilimit))
dsnoirr_newtime_GAR_cut_filter_lai = po_noirri_GAR_split.where((po_noirri_SGAR.IRRIFRAC>irrilimit))

po_irri_SGAR_filter_lai = po_irri_SGAR.where((po_irri_SGAR.IRRIFRAC>irrilimit))
po_noirri_SGAR_filter_lai = po_noirri_SGAR.where((po_noirri_SGAR.IRRIFRAC>irrilimit))

################### CALCULATE AND PLOT ###############################################

var_list = ["EVAPIRR","TEMP2",  "RH" ]
varname_list = [ "evapotranspiration [mmd$^{-1}$]", "T2Mean [°C]","RH [%]" ]

fig_name = ['(a)', '(c)', '(e)']
fig_name_diff = ['(b)', '(d)', '(f)']
fig, axs = plt.subplots(3, 2, sharex = True, figsize=(20,11))

params = {
    "legend.fontsize": 13,
    "legend.markerscale": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
}

for i, (var, var_name) in enumerate(zip(var_list, varname_list)):
    if var == "TEMP2": 
        var_irri_day_GAR   =  dsirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_noirri_day_GAR =  dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_irri_day_SGAR   =  po_irri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_noirri_day_SGAR =  po_noirri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        ymax = 35
        ymin = -1
        ymax_diff = 5
        ymin_diff = -10
    elif var == "RH": 
        var_irri_day_GAR   =  dsirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_noirri_day_GAR =  dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_irri_day_SGAR   =  po_irri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_noirri_day_SGAR =  po_noirri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        ymax = 100
        ymin = 0
        ymax_diff = 50
        ymin_diff = -50
    elif var == "EVAPIRR": 
        var_irri_day_GAR   =  -(dsirr_newtime_GAR_cut_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_irri_day_GAR   =  var_irri_day_GAR.where(var_irri_day_GAR!=0).mean(['rlon','rlat'])
        var_noirri_day_GAR =  -(dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_noirri_day_GAR =  var_noirri_day_GAR.where(var_noirri_day_GAR!=0).mean(['rlon','rlat'])      
        var_irri_day_SGAR   =  -(po_irri_SGAR_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_irri_day_SGAR   =  var_irri_day_SGAR.where(var_irri_day_SGAR!=0).mean(['rlon','rlat'])
        var_noirri_day_SGAR =  -(po_noirri_SGAR_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_noirri_day_SGAR =  var_noirri_day_SGAR.where(var_noirri_day_SGAR!=0).mean(['rlon','rlat'])
        ymax = 12
        ymin = -1
        ymax_diff = 10
        ymin_diff = -5
    else: 
        print('Variable not in the list.')

    # diff 
    var_diff_GAR = var_irri_day_GAR - var_noirri_day_GAR
    var_diff_SGAR = var_irri_day_SGAR - var_noirri_day_SGAR

    a_1 = var_irri_day_GAR.values
    a_2 = var_irri_day_SGAR.values
    b_1 = var_noirri_day_GAR.values
    b_2 = var_noirri_day_SGAR.values

    data = {
        "date": np.concatenate(
            (
                var_irri_day_GAR.time.values,
                var_irri_day_SGAR.time.values, 
                var_noirri_day_GAR.time.values,
                var_noirri_day_SGAR.time.values
            )
        ),
        "values": np.concatenate((a_1, a_2, b_1, b_2)),
        "simulation": ['irrigated'] * len(a_1)
        + ['irrigated'] * len(a_2)
        + ['not irrigated'] * len(b_1)
        + ['not irrigated'] * len(b_2),
        "resolution": ["0.11°"] * len(a_1)
        + ["0.0275°"] * len(a_2)
        + ["0.11°"] * len(b_1)
        + ["0.0275°"] * len(b_2),
       }

    colors = {'irrigated': "blue",
             'not irrigated': "darkorange",}  
    
    plt.rcParams.update(params)
  
    p = sns.lineplot(
        data=data,
        x="date",
        y="values",
        hue="simulation",
        style="resolution",
        palette=colors,
        ax=axs[i,0],
        linewidth = 1.0,
    )
    p.legend_.remove()
    p.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1]))
    axs[i,0].grid(True)
    axs[i,0].set_ylim(ymin, ymax)
    axs[i,0].set_ylabel(str(var_name))
    axs[i,0].tick_params(axis='x', labelrotation=45)
    axs[i,0].set_xlabel("")
    axs[i,0].text(0.0, 1.08, fig_name[i], transform=axs[i,0].transAxes, fontsize=14)

    if var == 'EVAPIRR':
        p.legend(bbox_to_anchor=(1.0, 1.04))  
    
    # diff 
    a_1 = var_diff_GAR.values
    a_2 = var_diff_SGAR.values

    data = {
        "date": np.concatenate(
            (
                var_diff_GAR.time.values,
                var_diff_SGAR.time.values
            )
        ),
        "[°C]": np.concatenate((a_1, a_2)),
        "resolution": ["0.11°"] * len(a_1)
        + ["0.0275°"] * len(a_2),
       }

    colors = {'0.11°': "black",
              '0.0275°': "black",
             }      

    p = sns.lineplot(
        data=data,
        x="date",
        y="[°C]",
        hue="resolution",
        style="resolution",
        palette=colors,
        ax=axs[i,1],
        linewidth = 1.0,
    )
    p.legend_.remove()
    p.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1]))
    axs[i,1].grid(True)
    axs[i,1].set_ylim(ymin_diff,ymax_diff)
    axs[i,1].set_ylabel('Δ '+str(var_name))
    axs[i,1].tick_params(axis='x', labelrotation=45)
    axs[i,1].set_xlabel("")
    axs[i,1].text(0.0, 1.08, fig_name_diff[i], transform=axs[i,1].transAxes, fontsize=14)
    plt.tight_layout()
plt.subplots_adjust(hspace = 0.4, wspace = 0.4)    
plt.savefig(str(dir_out) + "/timeseries_temp_rh_evap.png", dpi=300, bbox_inches="tight")


####################### SICNIFICANCE TEST ##############################

def test_significance(dist1, dist2, alpha=0.05):
  
    if not isinstance(dist1, xr.DataArray) or not isinstance(dist2, xr.DataArray):
        raise TypeError("Both inputs must be xarray.DataArray objects.")

    # Convert DataArrays to NumPy arrays while aligning them on common dimensions
    #dist1, dist2 = xr.align(dist1, dist2, join="inner")  # Ensure they have the same shape
    # Convert to NumPy arrays
    dist1_values = dist1.values
    dist2_values = dist2.values

    # Clean the data by removing NaNs
    mask = ~np.isnan(dist1_values) & ~np.isnan(dist2_values)
    dist1_cleaned = dist1_values[mask]
    dist2_cleaned = dist2_values[mask]
    # Perform Shapiro-Wilk test for normality
    shapiro_dist1 = stats.shapiro(dist1_cleaned)
    shapiro_dist2 = stats.shapiro(dist2_cleaned)
    
    # Print Shapiro test results
    print(f"Shapiro test (dist1): p-value = {shapiro_dist1.pvalue:.4f}")
    print(f"Shapiro test (dist2): p-value = {shapiro_dist2.pvalue:.4f}")
    
    # Step 2: Choose the appropriate test based on normality results
    if shapiro_dist1.pvalue > 0.05 and shapiro_dist2.pvalue > 0.05:
        # If both distributions are normal, perform a paired t-test
        t_stat, p_value = stats.ttest_rel(dist1_cleaned, dist2_cleaned)
        print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    else:
        # If data is NOT normal, perform the Wilcoxon Signed-Rank Test
        w_stat, w_p = stats.wilcoxon(dist1_cleaned, dist2_cleaned, alternative='two-sided')
        print(f"Wilcoxon Signed-Rank Test: statistic = {w_stat:.4f}, p-value = {w_p:.4f}")
        p_value = w_p
    
    # Step 3: Interpret significance based on p-value
    if p_value < 0.05:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is NOT statistically significant (p ≥ 0.05).")
    
    print('*********************************************************************')

# test in loop
var_list = ["EVAPIRR","TEMP2",  "RH" ]

for var in var_list:
    if var == "TEMP2": 
        var_irri_day_GAR   =  dsirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_noirri_day_GAR =  dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_irri_day_SGAR   =  po_irri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
        var_noirri_day_SGAR =  po_noirri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])-273.15
    elif var == "RH": 
        var_irri_day_GAR   =  dsirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_noirri_day_GAR =  dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_irri_day_SGAR   =  po_irri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
        var_noirri_day_SGAR =  po_noirri_SGAR_filter[var].resample(time="D").mean().squeeze(['height2m','pft_irr']).mean(['rlon','rlat'])
    elif var == "EVAPIRR": 
        var_irri_day_GAR   =  -(dsirr_newtime_GAR_cut_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_irri_day_GAR   =  var_irri_day_GAR.where(var_irri_day_GAR!=0).mean(['rlon','rlat'])
        var_noirri_day_GAR =  -(dsnoirr_newtime_GAR_cut_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_noirri_day_GAR =  var_noirri_day_GAR.where(var_noirri_day_GAR!=0).mean(['rlon','rlat'])
        var_irri_day_SGAR   =  -(po_irri_SGAR_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_irri_day_SGAR   =  var_irri_day_SGAR.where(var_irri_day_SGAR!=0).mean(['rlon','rlat'])
        var_noirri_day_SGAR =  -(po_noirri_SGAR_filter[var].resample(time="D").sum().squeeze(['pft_irr']))
        var_noirri_day_SGAR =  var_noirri_day_SGAR.where(var_noirri_day_SGAR!=0).mean(['rlon','rlat'])
    else: 
        print('Variable not in the list.')
        
    print(var)
    print('test  GAR IRRI vs. SGAR IRRI')
    test_significance( var_irri_day_GAR,  var_irri_day_SGAR)
    print('test  GAR NOIRRI vs. SGAR NOIRRI')
    test_significance( var_noirri_day_GAR,  var_noirri_day_SGAR)
    print('test  GAR IRRI vs. GAR NOIRRI')
    test_significance(var_irri_day_GAR,  var_noirri_day_GAR)
    print('test  SGAR IRRI vs. SGAR NOIRRI')
    test_significance(var_irri_day_SGAR,  var_noirri_day_SGAR)






