zeus2_toolbox is a python package developed for reducing and analyzing data taken by the Second-generation z (Redshift)
and Early Universe Spectrometer (
ZEUS2, [Ferkinhoff, C. et al. ApJ 780, 142, 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...780..142F/abstract)).

Because the data acquired with the MCE does not represent the actual TES layout on the detector array, the data needs to
be trimmed and mapped according to an "array map" before it can be properly processed. Besides, due to various types of
noise present in the data, there is a strong need for the capability to perform high level analysis of the data. Hence,
I developed this package which enables me to read, write, visualize and do arithmetic or more sophisticated operations
on both the detector and ancillary data.

The package is composed of four submodules

- **tools** containing many small helper functions
- **zeus2io** containing the classes that can read and write both MCE and ancillary data
    - ArrayMap: the class that contains the array map information, can read in array map stored in CSV file
    - Obs: the class hosting the raw MCE, and optionally many ancillary data including Chop, TimeStamp, ObsId, ObsInfo
    - ObsArray: the class with the data in Obs rearranged according to a given ArrayMap
- **view** containing the classes to visualize data in various ways
    - FigFlux: the class that can plot the flux on a 2d map
    - FigArray: the class that can plot the data of each pixel in either detector or MCE layout
    - FigSpec: the class to plot the spectrum of the detector array
- **pipeline** containing the functions needed for various levels of data reduction

Documentation
---------------

The documentation can be found
at [https://zeus2-toolbox.readthedocs.io/](https://zeus2-toolbox.readthedocs.io/en/stable/). It currently contains only
the API, but a tutorial with examples and jupyter notebook will be added.

Example scripts and jupyter notebooks are also included in the "example" folder.

Acknowledgements
----------------

The project is based on the previous pipeline data reduction codes developed by Cody Lamarche and
the [zeustools](https://github.com/NanoExplorer/zeustools) project by Christopher Rooney. The mce_data submodule is
copied
from [multi-channel-electronics / mce_script](https://github.com/multi-channel-electronics/mce_script/tree/master/python)
.
The real_unit() function is copied
from [zeustools/iv_tools](https://github.com/NanoExplorer/zeustools/blob/master/zeustools/iv_tools.py).
