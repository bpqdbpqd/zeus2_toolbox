"""
This is a template script to combine the weblog table, match with all the MCE
data files in DATA_DIR and the sub-folders, as well as the information in the
.hk and .run files. The matched table will be saved as ``matched_log.csv''.
"""

from zeus2_toolbox import pipeline as z2pipl

# ============================== configuration ==================================

OBS_LOG_DIR =  # path to the folder containing the APEX html observation log
# files, leave None if you don't need/have obs logs

DATA_DIR =  # path to the root data folder, which should contain multiple
# sub-folders names as "YYYYMMDD" where the data files are placed
WRITE_DIR = None  # path to the folder to save the performance result like
# figures or tables, leave None if you want to use the current folder

TABLE_SAVE = True  # flag whether to save the matched log as csv table

# ============================ run the pipeline ================================

obs_log = z2pipl.ObsLog.read_folder(OBS_LOG_DIR)
obs_info = z2pipl.ObsInfo()

f_list, beam_header_list = z2pipl.os.listdir(DATA_DIR), []
for fname in f_list:
    if z2pipl.os.path.isdir(z2pipl.os.path.join(DATA_DIR, fname)):
        fd_f_list = z2pipl.os.listdir(z2pipl.os.path.join(DATA_DIR, fname))
        for fd_fname in fd_f_list:
            if ("%s.hk" % fd_fname in fd_f_list) and \
                    ("%s.ts" % fd_fname in fd_f_list):
                beam_header_list.append(
                        z2pipl.os.path.join(DATA_DIR, fname, fd_fname))
    elif ("%s.hk" % fname in f_list) and \
            ("%s.ts" % fname in f_list):
        beam_header_list.append(z2pipl.os.path.join(DATA_DIR, fname))

for beam_header in beam_header_list:
    try:
        beam = z2pipl.Obs.read_header(
                filename=beam_header, try_data=False, try_chop=False, try_ts=True,
                try_info=True)
        beam.match_obs_log(obs_log)
        obs_info.append(beam.obs_info_)
    except Exception as err:
        z2pipl.warnings.warn("Failed to add the info for %s due to %s: %s" %
                             (beam_header, type(err), err), UserWarning)

obs_info.table_.sort("CTIME")
if TABLE_SAVE:
    obs_info.table_.write(
            z2pipl.os.path.join("" if WRITE_DIR is None else WRITE_DIR,
                                "matched_log.csv"), overwrite=True)
