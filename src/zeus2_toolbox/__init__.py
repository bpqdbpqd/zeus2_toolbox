# @Date    : 2021-01-29 16:26:51
# @Credit  : Bo Peng(bp392@cornell.edu), Cody Lamarche, Christopher Rooney
# @Name    : __init__.py
# @Version : 2.0
"""
zeus2_toolbox is a python package developed for processing the data acquired by
the ZEUS2 instrument (`Ferkinhoff, C. et al. ApJ 780, 142, 2014
<https://ui.adsabs.harvard.edu/abs/2014ApJ...780..142F/abstract>`_). The code
is available at `GitHub <https://github.com/bpqdbpqd/zeus2_toolbox.git>`_, and
the full documentation can be found at
`Read the Docs <https://zeus2-toolbox.readthedocs.io>`_.

The package includes four submodules:
--------------
tools
    basic but useful helper functions to process data independence of data
    structure
io
    io and data structure to deal with actual data
view
    figure classes to visualize zeus2 data
iv
    converting dac unit to physical units, and dealing with bias ramp data
pipeline
    wrapped and read-for-use functions to process zeus2 data
"""

from . import tools
from . import io
from . import convert
from . import view
from . import pipeline
