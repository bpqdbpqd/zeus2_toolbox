1.5

- Add cross flag to reduce_beam(), reduce_beams(), reduce_calibration() to accommodate cross scan
- Add flag use_hk in reduce_zobs and reduce_beam_pairs forcing using manual nodding phase, it will raise an error if no
  nodding phase is found
- Fix a bug dealing with np.nanmedian(aixs=None) with all nan data

1.4

- Add the missing dependence of beautifulsoup4
- Change the figures to be plotted with analyze_performance, and save obs_info table
- Fix a bug in calculating nfft and noverlap in pipeline.analyze_performance()
- Fix a bug with proc_along_axis() on empty ObsArray
- Fix a bug for get_chop_flux in beam with only one chop phase

1.3

- Add function double_nanmad_flag and method DataObj.get_double_nanmad_flag, which do MAD flagging twice to deal with
  the data with a sudden jump to flag less data
- Add function read_beams to read a large number of beams in parallel
- Add function analyze_performance to evaluate pixel performance based on rms and power spectrum; implement it in
  pipeline reduction functions
- Add function eval_performance to evaluate pixel performance based on rms and power spectrum, plot power spectrum,
  dynamical spectrum and the whole time series
- Change the criteria for selecting usable pixel for ICA
- Change the criteria for flagging time series outlier to flag different chop phase separately
- Fix a bug in rebuilding time stamp
- Fix a bug in desnaking to deal with the case that ref_pix is in pix_flag_list
- Fix a bug in FigFlux.imshow_flag() which doesn't accept empty list

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
