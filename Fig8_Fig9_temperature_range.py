#!/usr/bin/env python
# coding: utf-8

# Analysis of Extereme temperatures in 0.0275 vs. 0.11 
# Creation of Figure 8 and 9

import os

print('working directory',os.getcwd())

import numpy as np
import pandas as pd 
import scipy as sp
import xarray as xr
import seaborn as sns
import scipy.stats as stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from functions_correcting_time import correct_timedim
from functions_plotting import rotated_pole
from functions_reading_files import read_efiles_in_chunks 

# create plotting directory 
dir_working = os.getcwd()
dir_out = os.path.join(os.getcwd(), "Figures/")
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
print("Output directory is: ", dir_out)

######################### read data ###################################################
year = 2017
month = 0
#
exp_number_irri_SGAR = "067109"
exp_number_noirri_SGAR = "067108"
#
exp_number_irri_GAR = "067027"
exp_number_noirri_GAR = "067026"

varlist_SGAR = ["T2MAX", "TEMP2", "T2MIN"]
var_num_list_SGAR = ["201", "167", "202"]

varlist_GAR = ["T2MAX", "TEMP2", "T2MIN"]
var_num_list_GAR = ["201", "167", "202"]

# adapt your paths where the data is stored.
# Please also check the functions in functions_reading_files.py. 
data_path_SGAR_noirri = "/noirri0275"
data_path_SGAR_irri = "/irri0275"
data_path_GAR_noirri = "/noirri11"
data_path_GAR_irri = "/irri11"

# SGAR 
for var_SGAR, var_num_SGAR in zip(varlist_SGAR, var_num_list_SGAR):
    single_var_data_SGAR_irri = read_efiles_in_chunks(data_path_SGAR, var_SGAR, var_num_SGAR, exp_number_irri_SGAR, year, month, 100, 1, 100, 100)
    single_var_data_SGAR_noirri = read_efiles_in_chunks(data_path_SGAR, var_SGAR, var_num_SGAR, exp_number_noirri_SGAR, year, month, 100, 1, 100, 100)
   
    if var_SGAR == varlist_SGAR[0]:
        ds_var_irri_SGAR = single_var_data_SGAR_irri
        ds_var_noirri_SGAR = single_var_data_SGAR_noirri
    else:
        ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, single_var_data_SGAR_irri])
        ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, single_var_data_SGAR_noirri])
# GAR
for var_GAR, var_num_GAR in zip(varlist_GAR, var_num_list_GAR):
    single_var_data_GAR_irri = read_efiles_in_chunks(data_path_GAR, var_GAR, var_num_GAR, exp_number_irri_GAR, year, month, 100, 1, 100, 100)
    single_var_data_GAR_noirri = read_efiles_in_chunks(data_path_GAR, var_GAR, var_num_GAR, exp_number_noirri_GAR, year, month, 100, 1, 100, 100)
   
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
ds_var_irri_SGAR = xr.merge([ds_var_irri_SGAR, irrifrac_SGAR, lsm_SGAR])
ds_var_noirri_SGAR = xr.merge([ds_var_noirri_SGAR, irrifrac_SGAR, lsm_SGAR])
ds_var_irri_GAR = xr.merge([ds_var_irri_GAR, irrifrac_GAR, lsm_GAR])
ds_var_noirri_GAR = xr.merge([ds_var_noirri_GAR, irrifrac_GAR, lsm_GAR])

        
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
po_irri_GAR = dsirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_GAR = dsnoirr_newtime_GAR.isel(rlat=slice(50, 72), rlon=slice(70, 108))
po_noirri_SGAR = cut_same_area(dsnoirr_newtime_SGAR, po_noirri_GAR)
po_irri_SGAR = cut_same_area(dsirr_newtime_SGAR, po_irri_GAR)

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


######################### irrifrac change mask for later ###################################################

def create_irrifrac_change_mask(limit): 
    irrifrac_GAR_split = po_noirri_GAR_split.IRRIFRAC.where(po_noirri_GAR_split.IRRIFRAC>0)
    irrifrac_SGAR = po_noirri_SGAR.IRRIFRAC.where(po_noirri_SGAR.IRRIFRAC>0)
    
    irri_diff = irrifrac_SGAR - irrifrac_GAR_split
    
    # Create an initial mask with NaN
    mask = xr.full_like(irri_diff, fill_value=float("nan"))
    # Condition where irrigation fraction is valid (> 0 in both)
    valid_area = (irrifrac_SGAR > 0.05) & (irrifrac_GAR_split > 0.05)
    mask = mask.where(~((valid_area) & (irri_diff > limit)), 1)  # Set 1 where irri_diff > 0.4
    mask = mask.where(~((valid_area) & (irri_diff < -limit)), 0)  # Set 0 where irri_diff < -0.4
    return mask

irrifrac_change =create_irrifrac_change_mask(0.10)


######################### calculate differences in Dataframe for boxplot #############################################

irrilimit = po_irri_SGAR.IRRIFRAC

def create_1d_diff_df(irri_data, noirri_data, resolution):
    diff_df = pd.DataFrame()  

    t2max_irri = irri_data["T2MAX"].resample(time = '1D').max().squeeze('height2m')-273.15
    t2max_noirri = noirri_data["T2MAX"].resample(time = '1D').max().squeeze('height2m')-273.15
    t2max_diff = t2max_irri - t2max_noirri
    t2max_diff = t2max_diff.where(irrilimit > 0.7)

    temp2_irri = irri_data["TEMP2"].resample(time = '1D').mean().squeeze('height2m')-273.15
    temp2_noirri = noirri_data["TEMP2"].resample(time = '1D').mean().squeeze('height2m')-273.15
    temp2_diff = temp2_irri - temp2_noirri
    temp2_diff = temp2_diff.where(irrilimit > 0.7)
    
    t2min_irri = irri_data["T2MIN"].resample(time = '1D').min().squeeze('height2m')-273.15
    t2min_noirri = noirri_data["T2MIN"].resample(time = '1D').min().squeeze('height2m')-273.15
    t2min_diff = t2min_irri - t2min_noirri
    t2min_diff = t2min_diff.where(irrilimit > 0.7)

    # in Dataframe 
    diff_df = pd.DataFrame() 
    diff_df["resolution"] = [str(resolution)]*len(t2max_diff.rlon)*len(t2max_diff.rlat)*len(t2max_diff.time)
    diff_df["T2MAX"] = t2max_diff.values.reshape(len(t2max_diff.rlon)*len(t2max_diff.rlat)*len(t2max_diff.time))
    diff_df["T2MEAN"] = temp2_diff.values.reshape(len(temp2_diff.rlon)*len(temp2_diff.rlat)*len(temp2_diff.time))
    diff_df["T2MIN"] = t2min_diff.values.reshape(len(t2min_diff.rlon)*len(t2min_diff.rlat)*len(t2min_diff.time))

    return diff_df 

diff_df_0275 = create_1d_diff_df(po_irri_SGAR, po_noirri_SGAR, 0.0275)
diff_df_11 = create_1d_diff_df(po_irri_GAR_split, po_noirri_GAR_split, 0.11)

diff_df = pd.concat([diff_df_11, diff_df_0275], axis = 0)
del(diff_df_0275)
del(diff_df_11)

# differences of differences 
def calc_t2max_diff(irri_data, noirri_data):
    t2max_irri = irri_data["T2MAX"].resample(time = '1D').max().squeeze('height2m')-273.15
    t2max_irri_month = t2max_irri.groupby('time.month').mean('time')
    t2max_noirri = noirri_data["T2MAX"].resample(time = '1D').max().squeeze('height2m')-273.15
    t2max_noirri_month = t2max_noirri.groupby('time.month').mean('time')
    t2max_diff = t2max_irri_month - t2max_noirri_month
    return t2max_diff

def calc_t2min_diff(irri_data, noirri_data):
    t2min_irri = irri_data["T2MIN"].resample(time = '1D').min().squeeze('height2m')-273.15
    t2min_irri_month = t2min_irri.groupby('time.month').mean('time')
    t2min_noirri = noirri_data["T2MIN"].resample(time = '1D').min().squeeze('height2m')-273.15
    t2min_noirri_month = t2min_noirri.groupby('time.month').mean('time')
    t2min_diff = t2min_irri_month - t2min_noirri_month
    return t2min_diff

def calc_t2mean_diff(irri_data, noirri_data):
    t2mean_irri = irri_data["TEMP2"].resample(time = '1D').mean().squeeze('height2m')-273.15
    t2mean_irri_month = t2mean_irri.groupby('time.month').mean('time')
    t2mean_noirri = noirri_data["TEMP2"].resample(time = '1D').mean().squeeze('height2m')-273.15
    t2mean_noirri_month = t2mean_noirri.groupby('time.month').mean('time')
    t2mean_diff = t2mean_irri_month - t2mean_noirri_month
    return t2mean_diff

t2max_diff_11   = calc_t2max_diff(po_irri_GAR_split, po_noirri_GAR_split)
t2max_diff_0275 = calc_t2max_diff(po_irri_SGAR, po_noirri_SGAR)

t2min_diff_11   = calc_t2min_diff(po_irri_GAR_split, po_noirri_GAR_split)
t2min_diff_0275 = calc_t2min_diff(po_irri_SGAR, po_noirri_SGAR)

t2mean_diff_11   = calc_t2mean_diff(po_irri_GAR_split, po_noirri_GAR_split)
t2mean_diff_0275 = calc_t2mean_diff(po_irri_SGAR, po_noirri_SGAR)

diffdiff_t2max  = t2max_diff_0275 - t2max_diff_11
diffdiff_t2min  = t2min_diff_0275 - t2min_diff_11
diffdiff_t2mean = t2mean_diff_0275 - t2mean_diff_11

######################### plot boxplot and scatterplot ###################################################

gs = gridspec.GridSpec(2, 3)
gs = gridspec.GridSpec(8, 3)
fig = plt.figure(figsize = (8,7))
# boxplot calculation 
ax1 = fig.add_subplot(gs[0:4,0:3])

plot_df =pd.melt(diff_df, id_vars=['resolution'])

params = {
    "legend.fontsize": 12,
    "legend.markerscale": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)
sns.set(style="ticks")
bp = sns.boxplot( 
            x="variable",
            y="value",
            hue="resolution",
            data=plot_df,
            width=0.8,
            ax = ax1,
            whis = [5, 95],
            boxprops= {"linewidth":0.8},
           # whiskerprops= {"linewidth":0.8},
            flierprops={"marker": "x", "color":"grey", "alpha": 0.5, "markersize" : 4},
            medianprops={"color":"red"})
ax1.set_ylim(-12, 12)
ax1.set_ylabel('Δ 2 m temperature [K]')
ax1.set_xlabel('')
ax1.grid(axis='y')
ax1.text(0.0, 1.08, '(a)', transform=ax1.transAxes, fontsize=12)

hatches = [None, '///', None, '///', None, '///']
c = ['black', 'black', 'black', 'black', 'black', 'black']
patches = [patch for patch in ax1.patches if type(patch) == mpl.patches.PathPatch]
# the number of patches should be evenly divisible by the number of hatches
h = hatches * (len(patches) // len(hatches))
# iterate through the patches for each subplot
for patch, hatch, color in zip(patches, h, c):
    patch.set_hatch(hatch)
    #fc = patch.get_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_facecolor('none')

l = ax1.legend()
    
for lp, hatch, color in zip(l.get_patches(), hatches, c):
    lp.set_hatch(hatch)
    fc = lp.get_facecolor()
    lp.set_edgecolor(color)
    lp.set_facecolor('none')


var_list = [ "T2MAX", "TEMP2" , "T2MIN"]
fig_name_list = ['(b)', '(c)','(d)']
           
label_GAR = '0.11°'
label_SGAR = '0.0275°'
ax_shared = None  # Initialize shared axis reference

for v, (var, fig_name) in enumerate(zip(var_list, fig_name_list)): 
    plt.rcParams.update(params)
    if var == "TEMP2":
        t2_diff_GAR   = calc_t2mean_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR   = calc_t2mean_diff(po_irri_SGAR, po_noirri_SGAR)
        ymin_diff = -10
        ymax_diff = 3
    elif var == "T2MAX":
        t2_diff_GAR   = calc_t2max_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR  = calc_t2max_diff(po_irri_SGAR, po_noirri_SGAR)
        ymin_diff = -10
        ymax_diff = 3
    elif var == "T2MIN":
        t2_diff_GAR   = calc_t2min_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR  = calc_t2min_diff(po_irri_SGAR, po_noirri_SGAR)
        ymin_diff = -10
        ymax_diff = 3
    else: 
        print('Variable not in var_list.')

    irrifrac_GAR    =  po_irri_GAR_split.IRRIFRAC
    irrifrac_SGAR   =  po_irri_SGAR.IRRIFRAC   
    # correlation
    df_GAR = pd.DataFrame()
    df_GAR['irrifrac_GAR']= irrifrac_GAR.values.flatten()
    df_GAR['t2_diff_GAR'] = t2_diff_GAR.values.flatten()
    df_GAR = df_GAR.dropna()
    r_GAR, p_GAR = sp.stats.pearsonr(df_GAR['irrifrac_GAR'], df_GAR['t2_diff_GAR'])

    df_SGAR = pd.DataFrame()
    df_SGAR['irrifrac_SGAR']= irrifrac_SGAR.values.flatten()
    df_SGAR['t2_diff_SGAR'] = t2_diff_SGAR.values.flatten()
    df_SGAR = df_SGAR.dropna()
    r_SGAR, p_SGAR = sp.stats.spearmanr(df_SGAR['irrifrac_SGAR'], df_SGAR['t2_diff_SGAR'])

    # scatter plot
    if ax_shared is None:
        ax2 = fig.add_subplot(gs[4:7, v])  # First subplot without sharing
        ax_shared = ax2  # Store the first subplot for sharing
    else:
        ax2 = fig.add_subplot(gs[4:7, v], sharey=ax_shared)  # Share y-axis with the first subplot
    ax2.grid(True, zorder=0)
    ax2.scatter(x=df_GAR['irrifrac_GAR'], y=df_GAR['t2_diff_GAR'], s= 5, marker='o',  alpha = 0.6, label=label_GAR , zorder = 6)
    ax2.scatter(x=df_SGAR['irrifrac_SGAR'], y=df_SGAR['t2_diff_SGAR'],s=5,  marker='o', alpha = 0.6, label = label_SGAR, zorder = 4)
    ax2.set_ylim(ymin_diff,ymax_diff)
    ax2.set_xlim(-0.1,1.01) 
    ax2.text(0.0, 1.08, fig_name, transform=ax2.transAxes, fontsize=12)
    ax2.text(0.03, 0.04, "Pearson's correlation \n0.11°    : r ={:.2f}".format(r_GAR)+" p={:.2f}".format(p_GAR)\
                  +"\n0.0275°: r ={:.2f}".format(r_SGAR)+" p={:.2f}".format(p_SGAR),\
                  transform=ax2.transAxes, fontsize=8, bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha = 0.7), zorder=6 )
    ax2.set_xlabel('irrifrac [-]')
    if v == 0: 
        ax2.set_ylabel('Δ 2 m temperature [K]')
        ax2.legend(fontsize=8, bbox_to_anchor=(1.02, 1.015), loc='upper right', markerscale=3)
        plt.setp(ax2.get_yticklabels(), visible=True)
    else: 
        plt.setp(ax2.get_yticklabels(), visible=False)

plt.tight_layout()
plt.subplots_adjust(hspace = 1.0, wspace = 0.2)    
#plt.savefig(str(dir_out) + "/combined_boxplot_scatter_corr_temp_irrifrac.png", dpi=300, bbox_inches="tight")


############################# significance test ###################################################

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



var_list = [ "T2MAX", "TEMP2" , "T2MIN"]

for v, (var, fig_name) in enumerate(zip(var_list, fig_name_list)): 
    plt.rcParams.update(params)
    if var == "TEMP2":
        t2_diff_GAR   = calc_t2mean_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR   = calc_t2mean_diff(po_irri_SGAR, po_noirri_SGAR)
        print('SIGNIFICANCE GAR vs SGAR', var)
        test_significance(t2_diff_GAR, t2_diff_SGAR)
    elif var == "T2MAX":
        t2_diff_GAR   = calc_t2max_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR  = calc_t2max_diff(po_irri_SGAR, po_noirri_SGAR)
        print('SIGNIFICANCE GAR vs SGAR', var)
        test_significance(t2_diff_GAR, t2_diff_SGAR)
    elif var == "T2MIN":
        t2_diff_GAR   = calc_t2min_diff(po_irri_GAR_split, po_noirri_GAR_split)
        t2_diff_SGAR  = calc_t2min_diff(po_irri_SGAR, po_noirri_SGAR)
        print('SIGNIFICANCE GAR vs SGAR', var)
        test_significance(t2_diff_GAR, t2_diff_SGAR)
    else: 
        print('Variable not in var_list.')

############################# spatial distributiont ###################################################

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


def plot_rotvar_adjust_cbar_vmin(
    fig,
    varvalue,
    axs,
    cbaxes,
    unit,
    label,
    cmap,
    levels,
    ticks,
    extend_scale,
    cbar,
    cbar_orient,
    vmin, 
    vmax
):
    axs.add_feature(land, zorder=1)
    axs.add_feature(coastline, edgecolor="black", linewidth=0.6)
    axs.add_feature(borders, edgecolor="black", linewidth=0.6)

    axs.set_xmargin(0)
    axs.set_ymargin(0)
    cmap = plt.get_cmap(cmap)
    rotplot = varvalue.plot.pcolormesh(
        ax=axs, cmap=cmap, levels=levels, extend=extend_scale, add_colorbar=False, vmin = vmin, vmax = vmax
    )
    if cbar:
        cbar=None
    else:
        # colorbar below the plot
        cbar = fig.colorbar(
            rotplot, cax=cbaxes, orientation=cbar_orient, label=label, ticks=ticks
        )
        cbar.cmap.set_under('blueviolet')
        cbar.cmap.set_over('deeppink')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.outline.set_visible(False)
    axs.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
    return rotplot

t2max_diff_11   = (calc_t2max_diff(po_irri_GAR_split, po_noirri_GAR_split)).where(po_noirri_GAR_split.BLA>0.5)
t2max_diff_0275 = (calc_t2max_diff(po_irri_SGAR, po_noirri_SGAR)).where(po_irri_SGAR.BLA>0.5)

t2min_diff_11   = (calc_t2min_diff(po_irri_GAR_split, po_noirri_GAR_split)).where(po_noirri_GAR_split.BLA>0.5)
t2min_diff_0275 = (calc_t2min_diff(po_irri_SGAR, po_noirri_SGAR)).where(po_irri_SGAR.BLA>0.5)

t2mean_diff_11   = (calc_t2mean_diff(po_irri_GAR_split, po_noirri_GAR_split)).where(po_noirri_GAR_split.BLA>0.5)
t2mean_diff_0275 = (calc_t2mean_diff(po_irri_SGAR, po_noirri_SGAR)).where(po_irri_SGAR.BLA>0.5)

diffdiff_t2max = t2max_diff_0275 - t2max_diff_11
diffdiff_t2min = t2min_diff_0275 - t2min_diff_11
diffdiff_t2mean = t2mean_diff_0275 - t2mean_diff_11


plot_diff_11_list   = [t2max_diff_11.sel(month = 6), t2mean_diff_11.sel(month = 6), t2min_diff_11.sel(month = 6)]
plot_diff_0275_list = [t2max_diff_0275.sel(month = 6), t2mean_diff_0275.sel(month = 6), t2min_diff_0275.sel(month = 6)]
plot_diffdiff_list  = [diffdiff_t2max.sel(month = 6), diffdiff_t2mean.sel(month = 6), diffdiff_t2min.sel(month = 6)]
title_list          = ['T2Max', 'T2Mean','T2Min']


fig_name_11    = ['(a)', '(d)', '(g)']
fig_name_0275  = ['(b)', '(e)', '(h)']
fig_name_diff  = ['(c)', '(f)', '(i)']

fig = plt.figure(figsize=(18,10))
spec = fig.add_gridspec(ncols=3, nrows=3)

cax1 = fig.add_axes([0.22, 0.08, 0.35, 0.02])
cax3 = fig.add_axes([0.678, 0.08, 0.22, 0.02])

levels_temp = list(filter(lambda num: num != 0,  np.arange(-4.5,5,0.5)))
ticks_temp =  np.arange(-4,5, 1)

levels_diff = list(filter(lambda num: num != 0,  np.arange(-2,2.25,0.25)))
ticks_diff =  np.arange(-2,3, 1)


for i, (plot_diff_11, plot_diff_0275, plot_diffdiff, title) in enumerate(zip(plot_diff_11_list, plot_diff_0275_list, plot_diffdiff_list, title_list)): 

    # 12 km 
    ax1 = fig.add_subplot(spec[0+i, 0], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar_vmin(fig, plot_diff_11, ax1, cax1, '°C', 'Δ T2 [K]', 'RdBu_r', levels_temp, ticks_temp, 'both', False, 'horizontal', levels_temp[0], levels_temp[-1] )
    ax1.set_title(title)
    ax1.text(0.035, 0.964, '0.11°', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax1.text(0.0, 1.05, fig_name_11[i], transform=ax1.transAxes, fontsize=14)

    
    # 3 km 
    ax2 = fig.add_subplot(spec[0+i, 1], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar_vmin(fig, plot_diff_0275, ax2, cax1, '°C', 'Δ T2 [K]', 'RdBu_r', levels_temp, ticks_temp, 'both', False, 'horizontal', levels_temp[0], levels_temp[-1])
    ax2.set_title(title)
    ax2.text(0.035, 0.964, '0.0275°', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax2.text(0.0, 1.05, fig_name_0275[i], transform=ax2.transAxes, fontsize=14)

    
    # diff 12 - 3 km
    ax3 = fig.add_subplot(spec[0+i, 2], projection=rotated_pole)
    rotplot = plot_rotvar_adjust_cbar(fig, plot_diffdiff, ax3, cax3, '°C', 'Δ T2 [K]', 'PiYG_r', levels_diff, ticks_diff, 'both', False, 'horizontal')
    ax3.set_title('*diff = 0.0275° - 0.11° (splitted)')
    ax3.text(0.035, 0.964, 'diff*', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=4.0), zorder=+6)
    ax3.text(0.0, 1.05, fig_name_diff[i], transform=ax3.transAxes, fontsize=14)
    # Overlay mask on the existing plot
    hatchplot = ax3.contourf(irrifrac_change['rlon'], irrifrac_change['rlat'], irrifrac_change, levels=[-0.5, 0.5, 1.5],hatches=['....', '/////'], colors='none', alpha=0, 
                zorder = 7)
        
plt.subplots_adjust(hspace = 0.3)
#plt.savefig(dir_out+'/extreme_temperature_spatial_new.png', dpi=300, bbox_inches='tight')



############################# irrifrac change mask ###################################################
# create mask for vertical profiles

sens = -0.5
mask_day = xr.where((diffdiff_t2max.squeeze('month') < sens) & (po_irri_SGAR.BLA > 0.5) & (po_irri_GAR_split.BLA > 0.5), 1, np.nan)
mask_day.plot()
mask_day.to_netcdf('mask_day.nc')

sens = 0.5
mask_night = xr.where((diffdiff_t2min.squeeze('month') > sens) & (po_irri_SGAR.BLA > 0.5) & (po_irri_GAR_split.BLA > 0.5), 1, np.nan)
mask_night.plot()
mask_night.to_netcdf('mask_night.nc')


