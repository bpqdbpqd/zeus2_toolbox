"""
An example script to get the ancillary information of the array, including the
wavelength [um], wavelength interval [um], sky transmission, the expected sky
power before the telescope dish [W] for each pixel, saved in .csv files.
"""

import os
from astropy.modeling import models
from astropy import units, constants

from zeus2_toolbox import pipeline as z2pipl

# =========================== reduction configuration ===========================

ARRAY_MAP_PATH =  # path to the array map file
BAND =  # choose the band you would like to use for the array map
ARRAY_CONF_PATH =  # path to the configuration file for the array
GRAT_IDX =  # grating index used
PWV =  # PWV value in mm
ELEV =  # elevation of the telescope in deg
TEMP =  # temperature of the sky in Kelvin
A_TELE =  # effective telescope area in m^2
OMEGA =  # pixel scale solid angle on sky in arcsec^2

WRITE_DIR = ""  # path to the folder to save the reduction result like figures
# or tables, leave "" if you want to use the current folder
WRITE_SUFFIX = ""  # suffix of the write name

# ========================= run the reduction pipeline =========================

array_map = z2pipl.ArrayMap.read(ARRAY_MAP_PATH)
array_map.read_conf(ARRAY_CONF_PATH)
array_map.set_band(BAND)
array_map.init_wl(grat_idx=GRAT_IDX)

# array wavelength
wl_arr, d_wl_arr = array_map.array_wl_, array_map.array_d_wl_
wl = z2pipl.ObsArray(arr_in=wl_arr[:, None], array_map=array_map)
d_wl = z2pipl.ObsArray(arr_in=d_wl_arr[:, None], array_map=array_map)
freq = z2pipl.wl_to_freq(wl)
d_freq = d_wl / wl * freq

# compute transmission
sky_trans_arr = z2pipl.transmission_pixel(
        freq=freq.data_, pwv=PWV, elev=ELEV, d_freq=d_freq.data_.mean())
sky_trans = z2pipl.ObsArray(arr_in=sky_trans_arr, array_map=array_map)

# power on sky without any telescope or instrument efficiency
sky_power_arr = (models.BlackBody(TEMP * units.K)(freq.data_ * units.GHz) *
                 (1 - sky_trans.data_) *
                 (d_freq.data_ * units.GHz) * (A_TELE * units.m ** 2) *
                 (OMEGA * units.arcsec ** 2)).to(units.W).to_value()
sky_power = z2pipl.ObsArray(arr_in=sky_power_arr, array_map=array_map)

wl.to_table().write(os.path.join(WRITE_DIR, "array_wl_%s%s.csv" %
                                 (BAND, WRITE_SUFFIX)))
d_wl.to_table().write(os.path.join(WRITE_DIR, "array_d_wl_%s%s.csv" %
                                   (BAND, WRITE_SUFFIX)))
sky_trans.to_table().write(os.path.join(WRITE_DIR, "sky_transmission_%s%s.csv" %
                                        (BAND, WRITE_SUFFIX)))
sky_power.to_table().write(os.path.join(WRITE_DIR, "sky_power_%s%s.csv" %
                                        (BAND, WRITE_SUFFIX)))
