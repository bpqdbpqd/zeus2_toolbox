1.2

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
