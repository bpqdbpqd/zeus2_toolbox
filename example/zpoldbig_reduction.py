"""
This is a template file which you can adapt to carry out the pipeline reduction
for the 5x5 raster data taken with the zpoldbig command.
"""

from zeus2_toolbox import pipeline as z2pipl

# ========================= reduction configuration =============================

ARRAY_MAP_PATH = None  # path to the array map file, leave None if you don't want
# to use any array map
OBS_LOG_DIR = None  # path to the folder containing the APEX html observation log
# files, leave None if you don't need/have obs logs
BAND = None  # choose the band you would like to use for the array map, the
# accepted values are 200, 350, 400 and 450, leave None if you
# want to use the whole array map

DATA_DIR =  # path to the folder containing the data
WRITE_DIR = None  # path to the folder to save the reduction result like figures
# or tables, leave None if you want to use the current folder

FLAT_HEADER =  # dictionary of the header and beam numbers of the flat field
# data, in the format of
# {'header1': [(start_beam1, end_beam1),
#              (start_beam2, end_beam2),
#              ...],
#  ...}
# e.g. {"skychop_191126": [(262, 263)]} combines
# skychop_191126_0262 and skychop_191126_0263 as the flat
# field, set to None if there is no flat data
DATA_HEADER =  # dictionary of the header and beam numbers of the science data
# in the same format as FLAT_HEADER

PARALLEL = True  # flag whether to run the reduction in parallel mode
TABLE_SAVE = True  # flag whether to save the reduction result as csv table
PLOT = True  # flag whether to plot the reduction result
PLOT_TS = True  # flag whether to plot the time series of each beam use in the
# reduction
REG_INTEREST = None  # the region of interest of the array to plot in the format
# of dictionary, e.g.
# REG_INTEREST={'spat_spec':[1, 11]} if you only want to
#  see the result of the pixel at [1, 11]
# REG_INTEREST={'spat_spec_list':[[1, 11], [1, 12]]} if you
#  want to see the result of a list of pixels
# REG_INTEREST={'spat':1} if you want to check all the
#  pixels at the spatial position 1
# REG_INTEREST={'spat_ran':[0, 2]} if you want to check
#  spatial position 0 through 2
# REG_INTEREST={'spat_ran':[0, 2], 'spec_ran':[6, 10]} will
#  show the result for all the pixels that are both in
#  spatial position range 0 to 2, and spectral index range
#  6 to 10
# leave None to plot the time series of all the pixels in
# the array, which can take a lot of time and slow down the
# reduction; please refer to the API document for
# ArrayMap.take_where() method for the accepted keywords
PLOT_FLUX = True  # flag whether to plot the flux of each beam
PLOT_SHOW = False  # flag whether to show the figures, can slow down the reduction
PLOT_SAVE = True  # flag whether to save the figures as png files

# ======================= run the reduction pipeline ===========================

array_map = z2pipl.ArrayMap.read(ARRAY_MAP_PATH)
if BAND is not None:
    array_map.set_band(BAND)
obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)

if FLAT_HEADER is not None:
    flat_result = z2pipl.reduce_skychop(
            flat_header=FLAT_HEADER, data_dir=DATA_DIR, write_dir=WRITE_DIR,
            array_map=array_map, obs_log=obs_log, parallel=PARALLEL,
            table_save=TABLE_SAVE, plot=PLOT, plot_ts=PLOT_TS,
            reg_interest=REG_INTEREST, plot_flux=PLOT_FLUX, plot_show=PLOT_SHOW,
            plot_save=PLOT_SAVE)
else:
    flat_result = (1, 0, [])
flat_flux, flat_err, pix_flag_list = flat_result[:2] + flat_result[-1:]

zpoldbig_result = z2pipl.reduce_zpoldbig(
        data_header=DATA_HEADER, data_dir=DATA_DIR, write_dir=WRITE_DIR,
        array_map=array_map, obs_log=obs_log, pix_flag_list=pix_flag_list,
        flat_flux=flat_flux, flat_err=flat_err, parallel=PARALLEL,
        return_pix_flag_list=True, table_save=TABLE_SAVE, plot=PLOT,
        plot_ts=PLOT_TS, reg_interest=REG_INTEREST, plot_flux=PLOT_FLUX,
        plot_show=PLOT_SHOW, plot_save=PLOT_SAVE)
zpoldbig_flux, zpoldbig_err, zpoldbig_pix_flag_list = zpoldbig_result[:2] + zpoldbig_result[-1:]
