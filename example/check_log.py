"""
This is a template script to combine the weblog table, match with all the MCE
data files in DATA_DIR and the sub-folders, as well as the information in the
.hk and .run files. The matched table will be saved as ``matched_log.csv''.
"""
import os.path
import warnings

import numpy as np

from zeus2_toolbox import pipeline as z2pipl

# ============================== configuration ==================================

OBS_LOG_DIR = None  # path to the folder containing the APEX html observation log
# files, leave None if you don't need/have obs logs
TEMP_PATH =  # path to the npy data storing the thermometry data

DATA_DIR =  # path to the root data folder, which should contain multiple
# sub-folders names as "YYYYMMDD" where the data files are placed
WRITE_DIR = None  # path to the folder to save the performance result like
# figures or tables, leave None if you want to use the current folder

TABLE_SAVE = True  # flag whether to save the matched log as csv table


# ============================ helper function  ================================

def list_data(fd):
    data_list = []
    if os.path.isdir(fd):
        flist = os.listdir(fd)
        for f in flist:
            if os.path.isdir(os.path.join(fd, f)) and \
                    not (len(f) >= 10 and f.isdecimal()):
                data_list += list_data(os.path.join(fd, f))
            elif (f + ".run") in flist:
                data_list += [os.path.join(fd, f)]

    return data_list


def list_setup(fd):
    setup_list = []
    if os.path.isdir(fd):
        flist = os.listdir(fd)
        for f in flist:
            if os.path.isdir(os.path.join(fd, f)):
                if len(f) >= 10 and f.isdecimal():
                    setup_list += [f]
                else:
                    setup_list += list_setup(os.path.join(fd, f))

    return setup_list


# ============================ run the pipeline ================================

if OBS_LOG_DIR is not None:
    obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)
else:
    obs_log = None
obs_info = z2pipl.ObsInfo()

beam_header_list = np.unique(list_data(DATA_DIR))
mce_setup_list = np.unique(list_setup(DATA_DIR))

print("Finished building beam file header list.")

for beam_header in beam_header_list:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore", message="The ObsLog object is empty.")
            warnings.filterwarnings(
                    "ignore", message="%s.ts not found" % beam_header)
            warnings.filterwarnings("ignore", message=
            ("Failed to read .hk for %s " % beam_header) +
            ("due to <class 'FileNotFoundError'>: %s" %
             ("%s or %s.hk are not hk files." % (beam_header, beam_header))))
            warnings.filterwarnings(
                    "ignore", message="No entry is found in obs log.")
            beam = z2pipl.Obs.read_header(
                    filename=beam_header, try_data=False, try_chop=False,
                    try_ts=True, try_info=True)
            beam.match_obs_log(obs_log)
        obs_info.append(beam.obs_info_)
    except Exception as err:
        z2pipl.warnings.warn("Failed to add the info for %s due to %s: %s" %
                             (beam_header, type(err), err), UserWarning)

print("Finished reading headers.")

setup_tb = z2pipl.Tb([["AUTOSETUP"] * len(mce_setup_list), mce_setup_list],
                     names=("obs_id", "CTIME"), masked=True)
obs_info.append(z2pipl.ObsInfo(tb_in=setup_tb))

time_tb = z2pipl.Tb([z2pipl.Time(, format = "unix").isot],
names = ("UTC_CTIME",), masked = True)
out_info = z2pipl.ObsInfo(tb_in=time_tb)
out_info.expand(obs_info)

if TEMP_PATH is not None:
    t, temp = np.load(TEMP_PATH).transpose()
    temp_ctime = z2pipl.naninterp(
            obs_info.table_["CTIME"].filled(np.nan).astype(float),
            t, temp, left=np.nan, right=np.nan)
    out_info.expand(z2pipl.ObsInfo(tb_in=z2pipl.Tb(
            [temp_ctime], names=("thermometer",), masked=True)))

out_info.table_.sort("CTIME")
if TABLE_SAVE:
    out_info.table_.write(
            z2pipl.os.path.join("" if WRITE_DIR is None else WRITE_DIR,
                                "matched_log.csv"), overwrite=True)
