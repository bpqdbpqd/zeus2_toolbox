"""
realtime noise analysis
"""

import datetime, time, os, numpy as np
from zeus2_toolbox import pipeline as z2pipl

# =========================== reduction configuration ===========================

ARRAY_MAP_PATH = None  # path to the array map file, leave None if you don't want
# to use any array map, leave it as None
OBS_LOG_DIR = None  # path to the folder containing the APEX html observation log
# files, leave None if you don't need/have obs logs
SIGN_PATH = None  # path to the table containing the sign information
BAND = None  # choose the band you would like to use for the array map, the
# accepted values are 200, 350, 400 and 450, leave None if you
# want to use the whole array map
CONF_PATH = None  # path to the array map configuration file

DATA_DIR =  # path to the folder containing the data
WRITE_DIR = None  # path to the folder to save the reduction result like figures
# or tables, leave None if you want to use the current folder

AVOID_HEADER_LIST = ["bias_step", "skychop", "iv"]  # will search in DATA_DIR
# for the .run files with the header not in the AVOID_HEADERS list

TABLE_SAVE = True  # flag whether to save the reduction result as csv table

PLOT = True  # flag whether to plot the reduction result
PLOT_TS = False  # by default not to plot folded and the original time series,
# which takes a long time
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
PLOT_PSD = False  # flag whether to plot power spectral diagram, SLOW!
PLOT_SPECGRAM = False  # flag whether to plot specgram, VERY SLOW!

PLOT_SHOW = False  # flag whether to show the figures, can slow down the reduction
PLOT_SAVE = True  # flag whether to save the figures as png files
ANALYZE = True  # flag whether to perform pixel performance analyze based on rms
# and power spectrum


################################### process ###################################

if ARRAY_MAP_PATH is not None:
    array_map = z2pipl.ArrayMap.read(ARRAY_MAP_PATH)
    if BAND is not None:
        array_map.set_band(BAND)
    if CONF_PATH is not None:
        array_map.read_conf(CONF_PATH)
    sign = z2pipl.ObsArray.read_table(SIGN_PATH).to_obs().to_obs_array(array_map)
else:
    array_map = None
    sign = z2pipl.ObsArray.read_table(SIGN_PATH).to_obs()
obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)

try:
    existing_set = set(os.listdir(DATA_DIR))
    print("Monitoring %s" % DATA_DIR)
except Exception as err:
    existing_set = set()
    print("Folder %s doesn't exist." % DATA_DIR)
queue_list, queue_size = [], []
queue_list_old, queue_size_old = [], []

while True:
    try:
        current_set = set(os.listdir(DATA_DIR))
    except Exception as err:
        current_set = set()
        if (datetime.datetime.now().minute % 10 == 0) and \
                (datetime.datetime.now().second < .6):
            print("Folder %s doesn't exist, standby." % DATA_DIR)

    new_file_set = current_set - existing_set
    if len(new_file_set) > 0:
        for fname in new_file_set:
            if (fname[-4:] == ".run") and not np.any(
                    [(header in fname) for header in AVOID_HEADER_LIST]):
                data_header = fname[:-4]
                if data_header not in queue_list:
                    queue_list.append(data_header)
                    data_path = os.path.getsize(
                            os.path.join(DATA_DIR, data_header))
                    if os.path.isfile(data_path):
                        queue_size.append(os.path.getsize(data_path))
                    else:
                        queue_size.append(0)

    time.sleep(0.5)

    if len(queue_list) > 0:
        print("Current in queue: %s" % queue_list)
        proc_list = []

        for q in queue_list[::-1]:
            if q in queue_list_old:
                queue_idx_old = np.where(np.asarray(queue_list_old) == q)[0][0]
                queue_idx = np.where(np.asarray(queue_list) == q)[0][0]
                path = os.path.join(DATA_DIR, q)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    if (size > 0) and (size == queue_size_old[queue_idx_old]):
                        proc_list.append(q)
                        queue_list.pop(queue_idx)
                        queue_size.pop(queue_idx)
                    else:
                        queue_size[queue_idx] = size

        if len(proc_list) > 0:
            print("Processing in queue: %s" % proc_list)

            for data_header in proc_list:
                data_result = z2pipl.eval_performance(
                        data_header=data_header, data_dir=DATA_DIR,
                        write_dir=WRITE_DIR, write_suffix="",
                        array_map=array_map, obs_log=obs_log, pix_flag_list=None,
                        parallel=False, return_ts=False, table_save=TABLE_SAVE,
                        plot=PLOT, plot_ts=PLOT_TS, reg_interest=REG_INTEREST,
                        plot_psd=PLOT_PSD, plot_specgram=PLOT_SPECGRAM,
                        plot_flux=PLOT_FLUX, plot_show=PLOT_SHOW, plot_save=PLOT_SAVE)
                print("Finished processing %s" % data_header)

            print("Finished queue")

    existing_set = current_set
    queue_list_old, queue_size_old = queue_list.copy(), queue_size.copy()
