"""
This is an example demonstrating how to reduce the science data for PLCK G244,
the data are plck_191128_0000 through 0039, and flat are skychop_191128_0068 and
0069
"""

from zeus2_toolbox import pipeline as z2pipl

# ========================= reduction configuration =============================

ARRAY_MAP_PATH = "../data/array_map_excel_alternative_20211203.csv"
# path to the array map file, leave None if you don't want to use any array map
OBS_LOG_DIR = "../data/all_apex_2019/apex_logs/obslogs"
# path to the folder containing the APEX html observation log files, leave None
# if you don't need/have obs logs
BAND = 400  # choose the band you would like to use for the array map, the
# accepted values are 200, 350, 400 and 450, leave None if you
# want to use the whole array map

DATA_DIR = "../data/all_apex_2019/20191128/"
# path to the folder containing the data
WRITE_DIR = "../nb/test"
# path to the folder to save the reduction result like figures or tables, leave
# None if you want to use the current folder

FLAT_HEADER = {"skychop_191128": [(68, 69)]}
# dictionary of the header and  beam numbers of the flat field data, in the format
# {'header1': [(start_beam1, end_beam1),
#              (start_beam2, end_beam2),
#              ...],
#  ...}
# e.g. {"skychop_191126": [(262, 263)]} combines
# skychop_191126_0262 and skychop_191126_0263 as the flat
# field, set to None if there is no flat data
DATA_HEADER = {"plck_191128": [(0, 39)]}
# dictionary of the header and beam numbers of the science data in the same
# format as FLAT_HEADER

DO_DESNAKE = False  # flag whether to perform desnaking
REF_PIX = None  # [spat_pos, spec_idx] of the reference pixel used to select
# other good pixels to build the snake model, e.g. [1, 11] means
# the pixel at spatial position 1 and spectral index 11 will be
# used as the reference, only matters if DO_DESNAKE=True
DO_SMOOTH = False  # flag whether to use a gaussian kernel to smooth the time
# series to remove the long term structure, an alternative
# de-trending process to desnaking
DO_ICA = True  # flag whether to use ICA decomposition to remove the correlated
# noise
SPAT_EXCL = [0, 2]  # list of the range of the spatial positions to be excluded
# from being used to build correlated noise model by ICA,
# should include at least +/- one spatial position to the
# target, e.g. if the source is placed at spat_pos=1,
# SPAT_EXCL should be [0, 2], or even [0, 3] or broader range
# if it appears extended

PARALLEL = True  # flag whether to run the reduction in parallel mode
TABLE_SAVE = True  # flag whether to save the reduction result as csv table
SAVE_WL = True
SAVE_ATM = True
PLOT = True  # flag whether to plot the reduction result
PLOT_TS = True  # flag whether to plot the time series of each beam use in the
# reduction
PLOT_ATM = True
REG_INTEREST = {"spat_ran": (0, 2), "spec_ran": (9, 13)}
# the region of interest of the array to plot in the format of dictionary, e.g.
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
ANALYZE = True  # flag whether to perform pixel performance analyze based on rms

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

zobs_result = z2pipl.reduce_zobs(
        data_header=DATA_HEADER, data_dir=DATA_DIR, write_dir=WRITE_DIR,
        array_map=array_map, obs_log=obs_log, pix_flag_list=pix_flag_list,
        flat_flux=flat_flux, flat_err=flat_err, parallel=PARALLEL, stack=DO_ICA,
        do_desnake=DO_DESNAKE, ref_pix=REF_PIX, do_smooth=DO_SMOOTH,
        do_ica=DO_ICA, spat_excl=SPAT_EXCL, return_pix_flag_list=True,
        table_save=TABLE_SAVE, save_wl=SAVE_WL, save_atm=SAVE_ATM,
        plot=PLOT, plot_ts=PLOT_TS, plot_atm=PLOT_ATM,
        reg_interest=REG_INTEREST, plot_flux=PLOT_FLUX,
        plot_show=PLOT_SHOW, plot_save=PLOT_SAVE, analyze=ANALYZE)
zobs_flux, zobs_err, zobs_pix_flag_list = zobs_result[:2] + zobs_result[-1:]
