1.3

- Add function double_nanmad_flag and method DataObj.get_double_nanmad_flag(), which do MAD flagging twice to deal with
  the data with a sudden jump to flag less data
- Add function read_beams() to read a large number of beams in parallel
- Change the criteria for selecting usable pixel for ICA
- Change the criteria for flagging time series outlier to flag different chop phase separately
- Fix a bug in rebuilding time stamp
- Fix a bug in desnaking to deal with the case that ref_pix is in pix_flag_list

1.2

- Add example jupyter notebook for the pipeline reduction
- Add example pipeline reduction scripts in the example folder
- Change the function name proc_calibration, proc_zpold, proc_zpoldbig to reduce_calibration, reduce_zpold,
  reduce_zpoldbig
- Fix a bug in pipeline.auto_flag_ts() which chooses the wrong flagging threshold

1.1

- Generated documentations on [https://zeus2-toolbox.readthedocs.io/](https://zeus2-toolbox.readthedocs.io/en/latest/)
- Add the ability to plot cumulative flux in pipeline.reduce_zobs()
- Add the ability to call FigArray.plot() in pipeline.plot_beam_ts()
- Fix a bug with multiprocessing.pool(), now generating pools with "Fork" context
- Suppress unnecessary warnings

1.0

- Initial release on GitHub
