#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:05:55 2023

@author: g300099
"""

import os

import xarray as xr
from functions_correcting_time import correct_timedim, correct_timedim_mfiles


def read_efiles(dir_path, var, var_num, exp_number, year, month):
    data_path = (
        str(dir_path)
    )
    print('data_path:', str(data_path)+"/var_series/"+ str(var))
    if month >= 1 and month <= 12:
        efiles = (
            "/var_series/"
            + str(var)
            + "/e"
            + str(exp_number)
            + "e_c"
            + str(var_num)
            + "_20170"
            + str(month)
            + ".nc"
        )
    else:
        efiles = (
            "/var_series/"
            + str(var)
            + "/e"
            + str(exp_number)
            + "e_c"
            + str(var_num)
            + "_2017*.nc"
        )
    data = xr.open_mfdataset(os.path.join(data_path, efiles))  # , parallel=True)
    return data


def read_efiles_in_chunks(dir_path, var, var_num, exp_number, year, month, chunk1, chunk2, chunk3, chunk4):
    data_path = (
        str(dir_path)
    )
    print('data_path:', str(data_path)+"/var_series/"+ str(var))
    if month >= 1 and month <= 12:
        efiles = (
            "/var_series/"
            + str(var)
            + "/e"
            + str(exp_number)
            + "e_c"
            + str(var_num)
            + "_20170"
            + str(month)
            + ".nc"
        )
    else:
        efiles = (
            "/var_series/"
            + str(var)
            + "/e"
            + str(exp_number)
            + "e_c"
            + str(var_num)
            + "_2017*.nc"
        )
    data = xr.open_mfdataset(os.path.join(data_path, efiles), chunks ={"time": chunk1,"lev":chunk2, "rlon":chunk3, "rlat": chunk4}, parallel=True, autoclose=True)
    return data

def cut_same_area(source_area, target_area):
    min_rlon_target=target_area.rlon[0].values 
    max_rlon_target=target_area.rlon[-1].values

    min_rlat_target=target_area.rlat[0].values 
    max_rlat_target=target_area.rlat[-1].values
    
    cutted_ds = source_area.sel(rlon=slice(min_rlon_target,max_rlon_target), \
                                rlat=slice(min_rlat_target,max_rlat_target))
    return cutted_ds
