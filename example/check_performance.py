"""
This is a template file which you can adapt to characterize the pixel performance
by measuring chop-wise rms, time series, power spectrum and dynamical spectrum.
The time taken to run the analysis on spectrosaurusrex is roughly 0.1s/pix/beam
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
WRITE_DIR = None  # path to the folder to save the performance result like
# figures or tables, leave None if you want to use the current folder

DATA_HEADER =  # dictionary of the header and beam numbers of the data, in the
# format of
# {'header1': [(start_beam1, end_beam1),
#              (start_beam2, end_beam2),
#              ...],
#  ...}
# e.g. {"w0533_191130": [(0, 183)], "orion_191130": [(0, 118)]} combines
# w0533_191130_0000 through 0183, with orion_191130_0000 through 118 as the
# input data

PIX_FLAG_LIST = []  # list of pixels to flag, [[spat1, spec1], [spat2, spec2], ...]
PARALLEL = True  # flag whether to read in data in parallel mode
TABLE_SAVE = True  # flag whether to save the average rms as csv table
PLOT = True  # flag whether to plot the performance result
PLOT_TS = True  # flag whether to plot the time series of all the beams
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
# CAUTIOUS: can slow down the program significantly if REG_INTEREST is too large
PLOT_PSD = True  # flag whether to plot power spectral diagram
PLOT_SPECGRAM = True  # flag whether to plot dynamical spectrum, can slow down
# the program significantly
PLOT_FLUX = True  # flag whether to plot the rms in 2-d array layout
PLOT_SHOW = False  # flag whether to show the figures, can slow down the reduction
PLOT_SAVE = True  # flag whether to save the figures as png files

# ======================= run the reduction pipeline ===========================

array_map = z2pipl.ArrayMap.read(ARRAY_MAP_PATH)
if BAND is not None:
    array_map.set_band(BAND)
obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)

check_result = z2pipl.eval_performance(
        data_header=DATA_HEADER, data_dir=DATA_DIR, write_dir=WRITE_DIR,
        array_map=array_map, obs_log=obs_log, pix_flag_list=PIX_FLAG_LIST,
        parallel=PARALLEL, table_save=TABLE_SAVE, plot=PLOT, plot_ts=PLOT_TS,
        reg_interest=REG_INTEREST, plot_psd=PLOT_PSD,
        plot_specgram=PLOT_SPECGRAM, plot_flux=PLOT_FLUX, plot_show=PLOT_SHOW,
        plot_save=PLOT_SAVE)
check_rms, check_pix_flag_list = \
    check_result[0], check_result[-1]
