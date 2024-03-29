from zeus2_toolbox.pipeline import *
from zeus2_toolbox.view import *

DATA_PATH = "../example_data/"
WRITE_PATH = ""
OBS_LOG_FD = "../example_data/obslog/"

CONF_PATH = "../example_data/grating_calibration_2021_apex.ini"

PLOT = True
PLOT_TS = True
PLOT_WHOLE_ARR = False
REG_INTEREST = {"spat_ran": (5, 8), "spec_ran": (5, 9)}
PLOT_FLUX = True
PLOT_SHOW = False
PLOT_SAVE = True
ANALYZE = True

ARRAY_MAP_PATH = "../example_data/array_200_excel.csv"

ARRAY_MAP_PATH = "../example_data/array_map_excel_alternative_20211203.csv"
BAND = 400
FLAT_HEADER = {"skychop_191127": [(34, 35)]}

FLAT_HEADER = {"skychop_211201": [(17, 17)]}
DATA_HEADER = {"ngc253_211201": [(0, 1)]}

REG_INTEREST = {"spat_ran": (0, 3)}

REF_PIX = [6, 7]
SPAT_EXCL = (0, 3)

array_map = ArrayMap.read(filename=ARRAY_MAP_PATH)  # read in array map
array_map.set_band(band=BAND)

if CONF_PATH is not None:
    array_map.read_conf(CONF_PATH)

obs_log = ObsLog.read_folder(folder=OBS_LOG_FD)  # read in obs_fft log
PARALLEL = True

flat_result = reduce_skychop(
        flat_header=FLAT_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
        array_map=array_map, obs_log=obs_log, pix_flag_list=[],
        parallel=PARALLEL, return_ts=False, return_pix_flag_list=True,
        table_save=True, plot=True, plot_ts=True, reg_interest=REG_INTEREST,
        plot_flux=True, plot_show=False, plot_save=True)
flat_flux, flat_err, flat_wt, pix_flag_list = flat_result

zobs_result = reduce_zobs(
        data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
        array_map=array_map, obs_log=obs_log, pix_flag_list=pix_flag_list,
        flat_flux=flat_flux, flat_err=flat_err, parallel=PARALLEL, stack=True,
        do_desnake=False, ref_pix=REF_PIX, do_smooth=False, do_ica=True,
        spat_excl=[0, 3], return_ts=False, return_pix_flag_list=False,
        table_save=True, plot=True, plot_ts=True,
        reg_interest=REG_INTEREST, plot_flux=PLOT_FLUX, plot_show=False,
        plot_save=PLOT_SAVE, analyze=ANALYZE, use_hk=False)

# ARRAY_MAP_PATH = "../example_data/array_map_excel_alternative_20211101.csv"
# BAND = 400
# FLAT_HEADER = {"skychop_191128": [(64, 64)]}
# DATA_HEADER = {"irc10216_191128": [(27, 51)]}
#
# array_map = ArrayMap.read(filename=ARRAY_MAP_PATH)  # read in array map
# array_map.set_band(band=BAND)
#
# flat_result = reduce_skychop(
#         flat_header=FLAT_HEADER,  data_dir=DATA_PATH, write_dir=WRITE_PATH,
#         array_map=array_map, obs_log=obs_log, pix_flag_list=[],
#         parallel=PARALLEL, return_ts=False, return_pix_flag_list=True,
#         table_save=False, plot=False, plot_ts=False, reg_interest=REG_INTEREST,
#         plot_flux=True, plot_show=False, plot_save=True)
# flat_flux, flat_err, flat_wt, pix_flag_list = flat_result

# calibration_result = reduce_calibration(
#         data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
#         array_map=array_map, obs_log=obs_log, is_flat=False,
#         pix_flag_list=pix_flag_list, flat_flux=flat_flux, flat_err=flat_err, cross=True,
#         parallel=PARALLEL, do_desnake=False, ref_pix=[1, 11],
#         do_smooth=False, do_ica=False, spat_excl=None, return_ts=False,
#         return_pix_flag_list=True, table_save=True, plot=True,
#         plot_ts=True, reg_interest=None, plot_flux=True, plot_show=False,
#         plot_save=True, analyze=False)

# zpoldbig_result = reduce_zpoldbig(
#         data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
#         array_map=array_map, obs_log=obs_log, is_flat=False,
#         pix_flag_list=pix_flag_list, flat_flux=flat_flux, flat_err=flat_err,
#         parallel=PARALLEL, do_desnake=False, ref_pix=[1, 11], do_smooth=True,
#         do_ica=False, spat_excl=(0, 5), return_ts=False,
#         return_pix_flag_list=True, table_save=True, plot=True, plot_ts=False,
#         reg_interest=None, plot_flux=True, plot_show=False, plot_save=True,
#         analyze=ANALYZE)

# eval_result = eval_performance(
#         data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
#         array_map=array_map, obs_log=obs_log, pix_flag_list=[],
#         parallel=PARALLEL, return_ts=False, table_save=True, plot=True,
#         plot_ts=True, reg_interest=REG_INTEREST, plot_psd=True, plot_specgram=True,
#         plot_flux=True, plot_show=False, plot_save=True)

# TODO: change to flag flux - flux_median < SNR
