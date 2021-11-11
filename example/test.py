from zeus2_toolbox.pipeline import *
from zeus2_toolbox.view import *


DATA_PATH = "../example_data/"
OBS_LOG_FD = "../example_data/obslog/"
ARRAY_MAP_PATH = "../../ref/array_200_excel.csv"
WRITE_PATH = ""

DATA_HEADER = {"orion_191127": [(95, 102)]}
FLAT_HEADER = {"skychop_191127": [(34, 35)]}
BAND = 200

PLOT = True
PLOT_TS = True
PLOT_WHOLE_ARR = False
REG_INTEREST = {"spat_ran": (6, 7), "spec_ran": (6, 8)}  # TODO: change
PLOT_FLUX = True
PLOT_SHOW = False
PLOT_SAVE = True

REF_PIX = [6, 7]
SPAT_EXCL = (5, 8)


array_map = ArrayMap.read(filename=ARRAY_MAP_PATH)  # read in array map
array_map.set_band(band=BAND)

obs_log = ObsLog.read_folder(folder=OBS_LOG_FD)  # read in obs_fft log

flat_result = reduce_skychop(
        flat_header=FLAT_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
        array_map=array_map, obs_log=obs_log, pix_flag_list=[],
        parallel=True, return_ts=False, return_pix_flag_list=True,
        table_save=True, plot=True, plot_ts=True, reg_interest=REG_INTEREST,
        plot_flux=True, plot_show=False, plot_save=True)
flat_flux, flat_err, flat_wt, pix_flag_list = flat_result

# zobs_result = reduce_zobs(
#         data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
#         array_map=array_map, obs_log=obs_log, pix_flag_list=pix_flag_list,
#         flat_flux=flat_flux, flat_err=flat_err, parallel=True, stack=True,
#         do_desnake=False, ref_pix=REF_PIX, do_smooth=False, do_ica=True,
#         spat_excl=SPAT_EXCL, return_ts=False, return_pix_flag_list=False,
#         table_save=True, plot=True, plot_ts=True,
#         reg_interest=REG_INTEREST, plot_flux=True, plot_show=False,
#         plot_save=True)

result = proc_calibration(
        data_header=DATA_HEADER, data_dir=DATA_PATH, write_dir=WRITE_PATH,
        array_map=array_map, obs_log=obs_log, pix_flag_list=pix_flag_list,
        flat_flux=flat_flux, flat_err=flat_err, parallel=True, do_desnake=False,
        ref_pix=REF_PIX, do_smooth=False, do_ica=False, spat_excl=SPAT_EXCL,
        return_ts=False, return_pix_flag_list=False, table_save=True, plot=True,
        plot_ts=False, reg_interest=REG_INTEREST, plot_flux=False, plot_show=False,
        plot_save=True)
