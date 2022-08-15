2.0

2.0 will be a major update adding the ability to map array to wavelength, plotting and correcting for transmission, and
converting data to physical unit.

- Update error prediction to 8/sqrt(3) * rms/sqrt(N) based on the data bandwidth 100 Hz
- Add method to ArrayMap class to read in grating configuration and converting array map to wavelength
- Add function freq_to_wl() and wl_to_freq()
- Add function transmission_raw(), transmission_raw_range(), transmission_smoothed(), transmission_smoothed_range(), and
  transmission_window() to pipeline
- Add set_xscale() and set_yscale() method to FigArray class
- Add class method ObsInfo.read() to read .hk and .run at the same time
- Add xscale parameter to FigArray.psd(), enabling to plot log-log power spectral diagram, and change the behaviour of
  scale parameter
- Add method FigFlux.savefig() which can compress the file size by a factor of three
- Add the ability to append ObsInfo with the same column of different data type
- Add more logical operators to DataObj class
- Add an example script array_sky.py to save array wavelength, transmission and sky power to table
- Fix a bug in zeus2_io.proc_along_axis()
- Fix a bug in FigSpec with twin_axes
- Fix many bugs in tools
- Fix a bug in math operation in DataObj
- Fix a bug in ArrayMap.take_where()
- Change TableObj.append() to be more flexible in handling incompatible type columns
- Change TimsStamps.read() to be more robust
- Change db to 10*log10(power) instead of 20, because power instead of root power is used here
- Change the behaviour of ObsArray.take_by_array_map()

1.6

- Add scipy as a dependence
- Add beautifulsoup4 as a requirement
- Add function tools.gaussian_2d()
- Add function tools.spec_to_wl(), tools.wl_to_spec() and tools.wl_to_grat_idx() which map spectral index to wavelength
  and vice verse, and compute the grating index to place certain wavelength at given [spat, spec]
- Add the ability to output the predicted error in pipeline.zobs()
- Add the ability to fit raster in pipeline.make_raster() and stack_raster()
- Change error prediction from rms/sqrt(#data)*2 to rms/sqrt(#data)*4*sqrt(2/3) which accounts for oversampling
- Fix a bug in FigFlux.write_text() in the position to put text for ObsArray
- Fix a bug in pipeline.reduce_beam_pair() that do_smooth is not propagated through
- Fix a bug in pipeline.reduce_zobs() by multiplying input times series by a factor of sqrt(2) if the time series is
  stacked

1.5

- Add make_raster() to turn beams_flux into raster shape and make figures; change reduce_zpold() to use make_raster()
- Add cross flag to reduce_beam(), reduce_beams(), reduce_calibration() to accommodate cross scan
- Add flag use_hk in reduce_zobs and reduce_beam_pairs forcing using manual nodding phase, it will raise an error if no
  nodding phase is found
- Change make_raster() so that it can accept nodding observation
- Change reduce_beam_pairs() and reduce_zobs, so that NOD_PHASE is moved to be applied in reduce_beam_pairs()
- Change the way multiple skychop are averaged, use nanmean instead of weighted mean
- Fix a bug dealing with np.nanmedian(axis=None) with all nan data
- Fix a bug in Obs.naninterp()
- Fix a bug in zeus2_io.fft_obs()

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
