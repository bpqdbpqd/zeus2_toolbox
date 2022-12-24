"""
realtime bias step processing
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

DATA_DIR =  # path to the folder containing the data
WRITE_DIR = None  # path to the folder to save the reduction result like figures
# or tables, leave None if you want to use the current folder

BS_HEADER = "bias_step"  # will search in DATA_DIR for the files with the header

DO_SMOOTH = True  # by default smoothing to remove large trend

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

PLOT_SHOW = False  # flag whether to show the figures, can slow down the reduction
PLOT_SAVE = True  # flag whether to save the figures as png files
ANALYZE = True  # flag whether to perform pixel performance analyze based on rms
# and power spectrum


################################### process ###################################

if ARRAY_MAP_PATH is not None:
    array_map = z2pipl.ArrayMap.read(ARRAY_MAP_PATH)
    if BAND is not None:
        array_map.set_band(BAND)
    sign = z2pipl.ObsArray.read_table(SIGN_PATH).to_obs().to_obs_array(array_map)
else:
    array_map = None
    sign = z2pipl.ObsArray.read_table(SIGN_PATH).to_obs()
obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)

if os.path.isdir(DATA_DIR):
    if os.path.isfile(os.path.join(DATA_DIR, "current_data_name")):
        with open(os.path.join(DATA_DIR, "current_data_name")) as f:
            fdname = f.read().strip()
        existing_set = set(os.listdir(os.path.join(DATA_DIR, fdname)))
    else:
        print("current_data_name not found in %s" % DATA_DIR)
        existing_set = set(os.listdir(os.path.join(DATA_DIR)))
else:
    existing_set = set()
    print("Folder %s doesn't exist." % DATA_DIR)

queue_list, queue_size = [], []
queue_list_old, queue_size_old = [], []

while True:

    if os.path.isdir(DATA_DIR):
        fd = DATA_DIR
        wfd = WRITE_DIR
        if os.path.isfile(os.path.join(DATA_DIR, "current_data_name")):
            with open(os.path.join(DATA_DIR, "current_data_name")) as f:
                fdname = f.read().strip()
            if os.path.isdir(os.path.join(DATA_DIR, fdname)):
                fd = os.path.join(DATA_DIR, fdname)
                current_set = set(os.listdir(fd))
                wfd = os.path.join(WRITE_DIR, fdname)
                if not os.path.isdir(wfd):
                    os.mkdir(wfd)
                    print("created %s." % wfd)
            else:
                if (datetime.datetime.now().minute % 10 == 0) and \
                        (datetime.datetime.now().second < .6):
                    print("%s doesn't exist in %s." % (fdname, DATA_DIR))
        else:
            if (datetime.datetime.now().minute % 10 == 0) and \
                    (datetime.datetime.now().second < .6):
                print("current_data_name not found, watching %s." % fd)
        current_set = set(os.listdir(fd))
    else:
        fd = os.path.join(DATA_DIR)
        wfd = WRITE_DIR
        current_set = set()
        if (datetime.datetime.now().minute % 10 == 0) and \
                (datetime.datetime.now().second < .6):
            print("Folder %s doesn't exist, standby." % DATA_DIR)

    new_file_set = current_set - existing_set
    if len(new_file_set) > 0:
        for fname in new_file_set:
            if (BS_HEADER in fname) and (fname[-4:] == ".run"):
                bs_header = fname[:-4]
                if bs_header not in queue_list:
                    queue_list.append(bs_header)
                    bs_path = os.path.join(fd, bs_header)
                    if os.path.isfile(bs_path):
                        queue_size.append(os.path.getsize(bs_path))
                    else:
                        queue_size.append(0)

    time.sleep(1)

    if len(queue_list) > 0:
        print("Current in queue: %s" % queue_list)
        proc_list = []

        for q in queue_list[::-1]:
            if q in queue_list_old:
                queue_idx_old = np.where(np.asarray(queue_list_old) == q)[0][0]
                queue_idx = np.where(np.asarray(queue_list) == q)[0][0]
                path = os.path.join(fd, q)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    if (size > 0) and (size == queue_size_old[queue_idx_old]):
                        proc_list.append(q)
                        queue_list.pop(queue_idx)
                        queue_size.pop(queue_idx)
                    else:
                        queue_size[queue_idx] = size

        if len(proc_list) > 0:
            time.sleep(3)
            print("Processing in queue: %s" % proc_list)

            for bs_header in proc_list:
                bs_result = z2pipl.reduce_bias_step(
                        data_header=bs_header, data_dir=fd,
                        write_dir=wfd,
                        array_map=array_map, obs_log=obs_log, pix_flag_list=None,
                        sign=sign, do_smooth=DO_SMOOTH,
                        return_pix_flag_list=True, table_save=TABLE_SAVE, plot=PLOT,
                        plot_ts=PLOT_TS, reg_interest=REG_INTEREST,
                        plot_flux=PLOT_FLUX, plot_show=PLOT_SHOW, plot_save=PLOT_SAVE,
                        analyze=ANALYZE)
                print("Finished processing %s" % bs_header)

            print("Finished queue")

    existing_set = current_set
    queue_list_old, queue_size_old = queue_list.copy(), queue_size.copy()
