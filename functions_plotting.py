#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 08:54:54 2023

@author: g300099
"""

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


def plot_rotvar(
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
    cbar_orient,
):
    axs.add_feature(land, zorder=1)
    axs.add_feature(coastline, edgecolor="black", linewidth=0.6)
    axs.add_feature(borders, edgecolor="black", linewidth=0.6)
    axs.set_xmargin(0)
    axs.set_ymargin(0)
    cmap = plt.get_cmap(cmap)
    rotplot = varvalue.plot.pcolormesh(
        ax=axs, cmap=cmap, levels=levels, extend=extend_scale, add_colorbar=False
    )
    # colorbar below the plot
    cbar = fig.colorbar(
        rotplot, cax=cbaxes, orientation=cbar_orient, label=label, ticks=ticks
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.outline.set_visible(False)
    axs.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
    return rotplot


def plot_rotvar(
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
    cbar_orient,
):
    axs.add_feature(land, zorder=1)
    axs.add_feature(coastline, edgecolor="black", linewidth=0.6)
    axs.add_feature(borders, edgecolor="black", linewidth=0.6)
    axs.set_xmargin(0)
    axs.set_ymargin(0)
    cmap = plt.get_cmap(cmap)
    rotplot = varvalue.plot.pcolormesh(
        ax=axs, cmap=cmap, levels=levels, extend=extend_scale, add_colorbar=False
    )
    # colorbar below the plot
    cbar = fig.colorbar(
        rotplot, cax=cbaxes, orientation=cbar_orient, label=label, ticks=ticks
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.outline.set_visible(False)
    axs.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
    return rotplot


def plot_rotvar_adjust_cbar(
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
    cbar_orient
):
    axs.add_feature(land, zorder=1)
    axs.add_feature(coastline, edgecolor="black", linewidth=0.6)
    axs.add_feature(borders, edgecolor="black", linewidth=0.6)

    axs.set_xmargin(0)
    axs.set_ymargin(0)
    cmap = plt.get_cmap(cmap)
    rotplot = varvalue.plot.pcolormesh(
        ax=axs, cmap=cmap, levels=levels, extend=extend_scale, add_colorbar=False
    )
    if cbar:
        cbar=None
    else:
        # colorbar below the plot
        cbar = fig.colorbar(
            rotplot, cax=cbaxes, orientation=cbar_orient, label=label, ticks=ticks
        )
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.outline.set_visible(False)
    axs.gridlines(linewidth=0.7, color="gray", alpha=0.8, linestyle="--", zorder=3)
    return rotplot
 

rotated_pole = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)


def draw_rectangel(background, x1, x2, y1, y2):
    x_Italy = [
        background.rlon[x1],
        background.rlon[x2],
        background.rlon[x2],
        background.rlon[x1],
        background.rlon[x1],
    ]
    y_Italy = [
        background.rlat[y1],
        background.rlat[y1],
        background.rlat[y2],
        background.rlat[y2],
        background.rlat[y1],
    ]
    return x_Italy, y_Italy



import pandas as pd 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_density_histogram(fig, ax2, data1_irri, data1_noirri, data2_irri, data2_noirri, data_obs, bins, legend, params):
    fig.suptitle('all months, all stations')
  
    ## precipitation    

    plt.rcParams.update(params)

    hist_plot = ax2.hist([\
              np.clip(data1_irri, bins[0], bins[-1]), \
              np.clip(data1_noirri, bins[0], bins[-1]),\
              np.clip(data2_irri, bins[0], bins[-1]), \
              np.clip(data2_noirri, bins[0], bins[-1]),\
              np.clip(data_obs, bins[0], bins[-1])],\
              bins=bins, histtype='bar', density = True) #, weights=weights)


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
    if legend == True: 
        # Add legends:
        sim = ['irri', 'noirri','obs']
        res = ['0.11°', '0.0275°']
        color_legend = ['cornflowerblue', 'darkorange','dimgrey']
        sim_legend = ax2.legend(
            [Line2D([0], [0], color=color, lw=4) for color in color_legend],
            sim, title='simulation', 
            loc = 'upper right') #bbox_to_anchor=(1.1, 1.0))

        res_legend = ax2.legend(
            [Patch(hatch=hatch, facecolor='white', edgecolor = 'black') for hatch in hatches[1::2]],
            res, bbox_to_anchor=(1., 0.77), 
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
    plt.xticks(bins)#, ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", ">=10", ""])
    plt.ylabel('density')
    return hist_plot

def plot_frequency_histogram(fig, ax2, data1_irri, data1_noirri, data2_irri, data2_noirri, data_obs, bins, legend, params):
    fig.suptitle('all months, all stations')
  
    ## precipitation    

    plt.rcParams.update(params)
    weights=[np.ones_like(data1_irri )*100 / len(data1_irri), \
             np.ones_like(data1_noirri)*100 / len(data1_noirri), \
             np.ones_like(data2_irri)*100 / len(data2_irri), \
             np.ones_like(data2_noirri)*100 / len(data2_noirri), \
             np.ones_like(data_obs)*100 / len(data_obs)]



    hist_plot = ax2.hist([\
              np.clip(data1_irri, bins[0], bins[-1]), \
              np.clip(data1_noirri, bins[0], bins[-1]),\
              np.clip(data2_irri, bins[0], bins[-1]), \
              np.clip(data2_noirri, bins[0], bins[-1]),\
              np.clip(data_obs, bins[0], bins[-1])],\
              bins=bins, histtype='bar', weights=weights)


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
    if legend == True: 
        # Add legends:
        sim = ['irri', 'noirri','obs']
        res = ['0.11°', '0.0275°']
        color_legend = ['cornflowerblue', 'darkorange','dimgrey']
        sim_legend = ax2.legend(
            [Line2D([0], [0], color=color, lw=4) for color in color_legend],
            sim, title='simulation', 
            loc = 'upper right') #bbox_to_anchor=(1.1, 1.0))

        res_legend = ax2.legend(
            [Patch(hatch=hatch, facecolor='white', edgecolor = 'black') for hatch in hatches[1::2]],
            res, loc = 'center right', #bbox_to_anchor=(1.1, 0.77), 
            title='resolution', labelspacing=.65)

        # for size of patch 
        for patch in sim_legend.get_patches():
            patch.set_height(10)
            patch.set_y(-3)

        ax2.add_artist(sim_legend)
    else: 
        pass
    plt.yscale("log")
    plt.ylabel('frequency [%]')
    # adjust plot 
    plt.xticks(bins)#, ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", ">=10", ""])
    return hist_plot