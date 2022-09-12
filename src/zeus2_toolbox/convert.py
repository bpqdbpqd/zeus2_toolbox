# @Date    : 2022-08-29 15:46:16
# @Credit  : Bo Peng(bp392@cornell.edu), Christopher Rooney
# @Name    : convert.py
# @Version : 2.0
"""
A module to perform several types of conversions. Based on the dac_converters
submodules of
`zeustools<https://github.com/NanoExplorer/zeustools/blob/master/zeustools>`_
package and will keep updated
"""

import warnings
import pkgutil

import numpy as np
from astropy import units, constants
from astropy.table import Table as Tb
from astropy.time import Time, TimeDelta
from scipy.signal import convolve

MCE_BIAS_R = 467  # ohm, MCE bias resistance, TODO: measure on the instrument
# dewar_bias_R = 49, Old value, not sure where it comes from?
DEWAR_BIAS_R = 130  # ohm, dewar bias resistance, old value 49

CMB_SHUNTS = (0, 3, 4)  # tuple of MCE columns with CMB_R shunt
ACTPOL_R = 180e-6  # ohm, ACTPOL MCE shunt resistance (830 nH)
CMB_R = 140e-6  # ohm, guessed value
DEWAR_FB_R = 5280  # ohm, uncertain

BUTTERWORTH_CONSTANT = 1217.9148  # When running in data mode 2 and the low pass
# filter is in the loop, all signals are multiplied by this factor

REL_FB_INDUCTANCE = 9  # This means that for a change of 1 uA in the
# TES, the squid will have to change 9 uA to keep up

MAX_BIAS_VOLTAGE = 5  # V, +- 5V
MAX_FB_VOLTAGE = 0.958  # V, uncertain

BIAS_DAC_BITS = 16
FB_DAC_BITS = 14

try:  # load sky transmission_raw data
    TRANS_TB = Tb.read(pkgutil.get_data(__name__, "resources/trans_data.csv").
                       decode("utf-8"), format="ascii.csv")
except Exception as err:
    warnings.warn("Failed to load transmission_raw data resource " +
                  "due to %s: %s" % (type(err), err), UserWarning)
    TRANS_TB = None


def bias_to_v_bias(bias, max_bias_voltage=MAX_BIAS_VOLTAGE,
                   bias_dac_bits=BIAS_DAC_BITS):
    """
    a helper function converting bias DAC value to bias voltage

    :param bias: scalar or array, tes bias value(s) in adc unit
    :type bias: int or float or numpy.ndarray
    :param int or float max_bias_voltage: scalar, maximum bias voltage in V
    :param int bias_dac_bits: int, bias DAC bit number
    :return: raw bias voltage in volt, in the same shape as input bias
    :rtype: int or float or numpy.ndarray
    """

    v_bias = bias / 2 ** bias_dac_bits * max_bias_voltage * 2
    # last factor of 2 is because voltage is bipolar

    return v_bias


def bias_to_i_bias(bias, mce_bias_r=MCE_BIAS_R, dewar_bias_r=DEWAR_BIAS_R,
                   max_bias_voltage=MAX_BIAS_VOLTAGE,
                   bias_dac_bits=BIAS_DAC_BITS):
    """
    a helper function converting bias DAC value to bias current

    :param bias: scalar or array, tes bias value(s) in adc unit
    :type bias: int or float or numpy.ndarray
    :param int or float mce_bias_r: scalar, MCE bias resistance in ohm
    :param int or float dewar_bias_r: scalar, dewar bias resistance in ohm
    :param int or float max_bias_voltage: scalar, maximum bias voltage in V
    :param int bias_dac_bits: int, bias DAC bit number
    :return: bias current in ampere, in the same shape as input bias
    :rtype: int or float or numpy.ndarray
    """

    v_bias = bias_to_v_bias(bias, max_bias_voltage=max_bias_voltage,
                            bias_dac_bits=bias_dac_bits)
    i_bias = v_bias / (dewar_bias_r + mce_bias_r)

    return i_bias


def fb_to_i_tes(fb, dewar_fb_r=DEWAR_FB_R, max_fb_voltage=MAX_FB_VOLTAGE,
                fb_dac_bits=FB_DAC_BITS,
                butterworth_constant=BUTTERWORTH_CONSTANT,
                rel_fb_inductance=REL_FB_INDUCTANCE):
    """
    a helper function converting feedback DAC values to the TES current

    :param fb: scalar or array, sq1 feedback value(s) in dac unit
    :type fb: int or float or numpy.ndarray
    :param int or float dewar_fb_r: scalar, dewar feedback resistance in ohm
    :param int or float max_fb_voltage: scalar, maximum feedback voltage in V
    :param int fb_dac_bits: int, feedback DAC bit number
    :param int or float butterworth_constant: scalar, accounting for the low pass
        filter
    :param int or float rel_fb_inductance: scalar, feedback inductance ratio
    :return: tes current in ampere, in the same shape as input fb
    :rtype: int or float or numpy.ndarray
    """

    fb_real_dac = fb / butterworth_constant
    v_fb = fb_real_dac / 2 ** fb_dac_bits * max_fb_voltage * 2
    # again, last factor of 2 is because voltage is bipolar
    i_fb = v_fb / dewar_fb_r
    i_tes = i_fb / rel_fb_inductance

    return i_tes


def fb_to_v_tes(bias, fb, mce_col=-1, mce_bias_r=MCE_BIAS_R,
                dewar_bias_r=DEWAR_BIAS_R, dewar_fb_r=DEWAR_FB_R,
                shunt_r=ACTPOL_R, alt_shunt_r=CMB_R, alt_col_list=CMB_SHUNTS,
                max_bias_voltage=MAX_BIAS_VOLTAGE, max_fb_voltage=MAX_FB_VOLTAGE,
                bias_dac_bits=BIAS_DAC_BITS, fb_dac_bits=FB_DAC_BITS,
                butterworth_constant=BUTTERWORTH_CONSTANT,
                rel_fb_inductance=REL_FB_INDUCTANCE):
    """
    a helper function converting feedback DAC values to TES voltage

    :param bias: scalar or array, tes bias value(s) in adc unit
    :type bias: int or float or numpy.ndarray
    :param fb: scalar or array, sq1 feedback value(s) in adc unit, must have the
        shape such that bias * fb * mce_col yields valid result
    :type fb: int or float or numpy.ndarray
    :param int or numpy.ndarray mce_col: int scalar or array, the MCE column
        number of the input data, because the resistor differs on a column base
        according to zeustools.iv_tools.real_units() description; must have the
        shape such that bias * fb * mce_col yields valid result; default -1 is
        not a physical column number, but it makes sure that the default
        shunt_r is used
    :param int or float mce_bias_r: scalar, MCE bias resistance in ohm
    :param int or float dewar_bias_r: scalar, dewar bias resistance in ohm
    :param int or float dewar_fb_r: scalar, dewar feedback resistance in ohm
    :param int or float shunt_r: scalar, the default shunt resistance in ohm used
        for data of MCE column not in the alt_col_list, default ACTPOL_R
    :param int or float alt_shunt_r: scalar, the alternative shunt resistance in
        ohm used for data of MCE column in the alt_col_list, default CMB_R
    :param tuple or list alt_col_list: list of MCE columns using the alternative
        shunt resistor, default CMB_SHUNT
    :param int or float max_bias_voltage: scalar, maximum bias voltage in V
    :param int or float max_fb_voltage: scalar, maximum feedback voltage in V
    :param int bias_dac_bits: int, bias DAC bit number
    :param int fb_dac_bits: int, feedback DAC bit number
    :param int or float butterworth_constant: scalar, accounting for the low pass
        filter
    :param int or float rel_fb_inductance: scalar, feedback inductance ratio
    :return: TES voltage in volt, in the shape of (bias * fb * mce_col)
    :rtype: int or float or numpy.ndarray
    """

    col = np.reshape(mce_col, newshape=(-1, 1))
    alt_col = np.reshape(alt_col_list, newshape=(1, -1))
    shunt_r_use = np.choose(np.any(col == alt_col, axis=1),
                            (shunt_r, alt_shunt_r))  # pick the shunt_r to use

    i_bias = bias_to_i_bias(
            bias=bias, mce_bias_r=mce_bias_r, dewar_bias_r=dewar_bias_r,
            max_bias_voltage=max_bias_voltage, bias_dac_bits=bias_dac_bits)
    i_tes = fb_to_i_tes(
            fb=fb, dewar_fb_r=dewar_fb_r, max_fb_voltage=max_fb_voltage,
            fb_dac_bits=fb_dac_bits, butterworth_constant=butterworth_constant,
            rel_fb_inductance=rel_fb_inductance)

    i_shunt = i_bias - i_tes
    v_tes = i_shunt * shunt_r_use

    return v_tes


def freq_to_wl(freq, freq_unit="GHz", wl_unit="um"):
    """
    a helper function converting frequency in unit specified in freq_unit to
    wavelength in wl_unit

    :param freq: float or array, value of the frequency to convert
    :type freq: float or numpy.ndarray
    :param str freq_unit: str, unit of input frequency, passed to
        astropy.units.Unit()
    :param str wl_unit: str, unit of output wavelength, passed to
        astropy.units.Unit()
    :return: float or array, wavelength in the same shape as input freq in the
        given wavelength unit
    :rtype: float or numpy.ndarray
    """

    wl = (constants.c / units.Unit(freq_unit)).to(units.Unit(wl_unit)). \
             to_value() / freq

    return wl


def wl_to_freq(wl, wl_unit="um", freq_unit="GHz"):
    """
    a helper function converting wavelength in unit specified in wl_unit to
    frequency in freq_unit

    :param wl: float or array, value of the wavelength to convert
    :type wl: float or numpy.ndarray
    :param str wl_unit: str, unit of input wavelength, passed to
        astropy.units.Unit()
    :param str freq_unit: str, unit of output frequency, passed to
        astropy.units.Unit()
    :return: float or array, frequency in the same shape as input wl in the
        given frequency unit
    :rtype: float or numpy.ndarray
    """

    freq = (constants.c / units.Unit(wl_unit)).to(units.Unit(freq_unit)). \
               to_value() / wl

    return freq


def spec_to_wl(spec, spat, grat_idx, order=5, rd=200, rg=0.711111111111111,
               px_shift=0, py_shift=0, alpha_min_index=73.9062453,
               c0=0.9800910119, c1=-0.0017009051035, c2=-0.87654448327,
               c3=36.248043521, c4=459.42373214, c5=-80.04474108,
               c6=-0.0017003774252, c7=-1.5498032937, c8=102.04705483,
               quad="spec"):
    """
    get the wavelength corresponding to the given grating index, spatial
    position, spectral index and the fitted parameters for ZEUS2; the default
    value for the fitted parameters are taken from
    zeus2_grating_calibration_apex2019 file; wavelength is calculated as

        wl = 5/order * (a (sin alpha_s)^2 + b * sin alpha_s + c)

    where alpha_s = alpha_min_index - grat_idx / (rd * rg) and
    theta = px + c6 * py**2 + c7 * py + c8 if quadratic term is spec, else
    py + c6 * px**2 + c7 * px + c8; px_shift and py_shift can shift
    lab calibration to fit with sky calibration

    :param spec: int or int array, spectral index used to compute wavelength; if
        input spec is an array, it must be compatible with spat so that
        spat+spec does not raise an error
    :type spec: int or numpy.ndarray
    :param spat: int or int array, spatial position considered; if input spat is
        an array, it must be compatible with spec so that spat+spec does not
        raise an error, and the output array shape will be the same as the
        shape of spat+spec array
    :type spat: int or numpy.ndarray
    :param int grat_idx: int, grating index
    :param int order: int, grating order to use
    :param rd: int or float, stepper motor, degrees per step
    :type rd: int or float
    :param float rg: float, grating degrees of movement per rotation of drive
        shaft
    :param px_shift: int or float, shift for spatial position to reconcile sky
        wavelength with fit parameters, will be added to spat to get px in theta
    :type px_shift: int or float
    :param py_shift: int or float, shift for spectral index to reconcile sky
        wavelength with fit parameters, will be added to spec to get py in theta
    :type py_shift: int or float
    :param float alpha_min_index: float, grating angle at the minimum index,
        computed from the reference alpha
    :param float c0: float, fitted coefficient
    :param float c1: float, fitted coefficient
    :param float c2: float, fitted coefficient
    :param float c3: float, fitted coefficient
    :param float c4: float, fitted coefficient
    :param float c5: float, fitted coefficient
    :param float c6: float, fitted coefficient
    :param float c7: float, fitted coefficient
    :param float c8: float, fitted coefficient
    :param str quad: str, allowed values are "spec" "spat", denoting which term
        is the quadratic term in the fit
    :return: float or array of the wavelength(es) for the input spat and spec
    :rtype: float or numpy.ndarray
    """

    alpha_s = alpha_min_index - grat_idx / (rd * rg)
    sin_alpha_s = np.sin(alpha_s * np.pi / 180)
    px, py = spat + px_shift, spec + py_shift
    theta = px + c6 * py ** 2 + c7 * py + c8 if "spec" in quad.lower() else \
        py + c6 * px ** 2 + c7 * px + c8

    a = c5
    b = c0 * theta + c4
    c = c1 * theta ** 2 + c2 * theta + c3
    wl = 5 / order * (a * sin_alpha_s ** 2 + b * sin_alpha_s + c)

    return wl


def wl_to_spec(wl, spat, grat_idx, order=5, rd=200, rg=0.711111111111111,
               px_shift=0, py_shift=0, alpha_min_index=73.9062453,
               c0=0.9800910119, c1=-0.0017009051035, c2=-0.87654448327,
               c3=36.248043521, c4=459.42373214, c5=-80.04474108,
               c6=-0.0017003774252, c7=-1.5498032937, c8=102.04705483,
               quad="spec"):
    """
    inverse function for grat_to_wl(), compute spectral index for input spatial
    position, wavelength at the given grat_idx

    :param wl: float or int array, wavelength used to compute spectral index; if
        input wl is an array, it must be compatible with spat so that spat+wl
        does not raise an error
    :type wl: float or numpy.ndarray
    :param spat: int or int array, spatial position considered; if input spat is
        an array, it must be compatible with spec so that spat+wl does not
        raise an error, and the output array shape will be the same as the
        shape of spat+spec array
    :type spat: int or numpy.ndarray
    :param int grat_idx: int, grating index
    :param int order: int, grating order to use
    :param rd: int or float, stepper motor, degrees per step
    :type rd: int or float
    :param float rg: float, grating degrees of movement per rotation of drive
        shaft
    :param px_shift: int or float, shift for spatial position to reconcile sky
        wavelength with fit parameters, will be added to spat to get px in theta
    :type px_shift: int or float
    :param py_shift: int or float, shift for spectral index to reconcile sky
        wavelength with fit parameters, will be added to spec to get py in theta
    :type py_shift: int or float
    :param float alpha_min_index: float, grating angle at the minimum index,
        computed from the reference alpha
    :param float c0: float, fitted coefficient
    :param float c1: float, fitted coefficient
    :param float c2: float, fitted coefficient
    :param float c3: float, fitted coefficient
    :param float c4: float, fitted coefficient
    :param float c5: float, fitted coefficient
    :param float c6: float, fitted coefficient
    :param float c7: float, fitted coefficient
    :param float c8: float, fitted coefficient
    :param str quad: str, allowed values are "spec" "spat", denoting which term
        is the quadratic term in the fit
    :return: float or array of the spectral index(es) for the input spat and wl
    :rtype: float or numpy.ndarray
    """

    alpha_s = alpha_min_index - grat_idx / (rd * rg)
    sin_alpha_s = np.sin(alpha_s * np.pi / 180)
    px = spat + px_shift

    a = c1
    b = c0 * sin_alpha_s + c2
    c = c3 + c4 * sin_alpha_s + c5 * sin_alpha_s ** 2 - wl * order / 5

    if "spec" in quad.lower():
        theta = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
        a = c6
        b = c7
        c = c8 + px - theta
        py = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
    else:
        theta = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
        py = theta - (c6 * px ** 2 + c7 * px + c8)
    spec = py - py_shift

    return spec


def wl_to_grat_idx(wl, spat, spec, order=5, rd=200, rg=0.711111111111111,
                   px_shift=0, py_shift=0, alpha_min_index=73.9062453,
                   c0=0.9800910119, c1=-0.0017009051035, c2=-0.87654448327,
                   c3=36.248043521, c4=459.42373214, c5=-80.04474108,
                   c6=-0.0017003774252, c7=-1.5498032937, c8=102.04705483,
                   quad="spec"):
    """
    compute the required grating index to place the input wavelength at the
    desired spatial position and spectral index

    :param wl: float or int array, wavelength used to compute grating index; if
        input wl is an array, it must be compatible with spat and spec so that
        wl+spat+spec does not raise an error, and the output array shape
        will be the same as the shape of wl+spat+spec array
    :type wl: float or numpy.ndarray
    :param spat: int or int array, spatial position used to compute grating
        index; if input spat is an array, it must be compatible with wl and spec
        so that wl+spat+spec does not raise an error
    :type spat: int or numpy.ndarray
    :param spec: int or int array, spectral index used to compute grating index;
        if input spec is an array, it must be compatible with wl and spat so
        that wl+spat+spec does not raise an error
    :type spec: int or numpy.ndarray
    :param int order: int, grating order to use
    :param rd: int or float, stepper motor, degrees per step
    :type rd: int or float
    :param float rg: float, grating degrees of movement per rotation of drive
        shaft
    :param px_shift: int or float, shift for spatial position to reconcile sky
        wavelength with fit parameters, will be added to spat to get px in theta
    :type px_shift: int or float
    :param py_shift: int or float, shift for spectral index to reconcile sky
        wavelength with fit parameters, will be added to spec to get py in theta
    :type py_shift: int or float
    :param float alpha_min_index: float, grating angle at the minimum index,
        computed from the reference alpha
    :param float c0: float, fitted coefficient
    :param float c1: float, fitted coefficient
    :param float c2: float, fitted coefficient
    :param float c3: float, fitted coefficient
    :param float c4: float, fitted coefficient
    :param float c5: float, fitted coefficient
    :param float c6: float, fitted coefficient
    :param float c7: float, fitted coefficient
    :param float c8: float, fitted coefficient
    :param str quad: str, allowed values are "spec" "spat", denoting which term
        is the quadratic term in the fit
    :return: float or array of the grating index(es) for the input spat, spec, wl
    :rtype: float or numpy.ndarray
    """

    px, py = spat + px_shift, spec + py_shift
    theta = px + c6 * py ** 2 + c7 * py + c8 if "spec" in quad.lower() else \
        py + c6 * px ** 2 + c7 * px + c8

    a = c5
    b = c0 * theta + c4
    c = c1 * theta ** 2 + c2 * theta + c3 - wl * order / 5
    alpha = np.arcsin((-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 / a) * 180 / np.pi
    grat_idx = -rd * rg * (alpha - alpha_min_index)

    return grat_idx


def transmission_raw_range(freq_ran, pwv, elev=60):
    """
    Compute transmission (curve) using the kappa and eta_0 data recorded in
    TRANS_TB, in the range of frequency specified in freq_ran. For the
    computation, please refer to test_transmission.ipynb notebook.

    :param freq_ran: list or tuple or array, the range of frequency in unit
        GHz to compute transmission, should be within the range [400, 1610); the
        largest and the smallest values will be interpreted as the range
    :type freq_ran: list or tuple or numpy.ndarray
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :return: (freq, trans), arrays recording the frequency and raw transmission
        computed in the given frequency range, at the input pwv and elevation
    :rtype: list
    :raises RuntimeError: transmission data not loaded
    """

    if TRANS_TB is None:  # check whether TRANS_TB is loaded
        raise RuntimeError("The resource of transmission_raw data is not loaded.")

    freq_min, freq_max = np.min(freq_ran), np.max(freq_ran)
    if np.any(freq_min < TRANS_TB["freq"].min()) or \
            np.any(freq_max > TRANS_TB["freq"].max()):  # check range
        warnings.warn("Input frequency out of the range.")

    flag_use = (freq_min - 0.10 < TRANS_TB["freq"]) & \
               (TRANS_TB["freq"] < freq_max + 0.10)
    freq_use = TRANS_TB["freq"][flag_use]
    trans_use = (TRANS_TB["eta0"] * np.exp(- pwv / np.sin(elev / 180 * np.pi) *
                                           TRANS_TB["kappa"]))[flag_use]

    trans_use[~np.isfinite(trans_use)] = 0

    return freq_use.data, trans_use.data


def transmission_raw(freq, pwv, elev=60):
    """
    Compute transmission (curve) at given pwv and elevation by calling
    transmission_raw_range, resampled at the input freq.

    :param freq: int or float or array, the frequency in unit GHz to compute
        transmission, should be within the range [400, 1610)
    :type freq: int or float or numpy.ndarray
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :return: variable or array recording the raw transmission computed at the
        given frequency, pwv and elevation, in the same shape as input freq
    :rtype: int or float or numpy.ndarray
    """

    freq_use, trans_use = transmission_raw_range(
            freq_ran=freq, pwv=pwv, elev=elev)
    trans = np.interp(freq, freq_use, trans_use)

    return trans


def gps_ts_to_time(ts):
    """
    Convert the time stamp recorded in .ts file to astropy.time.core.Time object,
    because the time stamps are in a very wierd format, such that it is using
    the same epoch as UTC, but timescale as GPS. This is probably because the
    computer calibrates the clock by GPS time, which is then converted some
    generic format like isot without considering the GPS-UTC offset, then this
    time is read in as if in UTC frame which is then converted to unix format.

    :param float or np.double ts: float or double, zeus2 time stamp
    :return: astropy.time.core.Time object in UTC frame
    :rtype: astropy.time.core.Time
    """

    ts_tai = Time(ts, format="unix", scale="tai") - \
             TimeDelta(19, format="sec", scale="tai")  # diff between GPS and TAI
    ts_utc = Time(ts_tai.unix, format="unix")

    return ts_utc


def time_to_gps_ts(time):
    """
    Convert the astropy.time.core.Time in UTC frame to zeus2 time stamp, which
    uses unix epoch but in GPS frame

    :param astropy.time.core.Time time: astropy.time.core.Time object in UTC
        frame
    :return: float or double, zeus2 like time stamp
    :rtype: float or np.double
    """

    ts_tai = Time(ts0, format="gps") + \
             TimeDelta(19, format="sec", scale="tai")  # diff between GPS and TAI
    ts = ts_tai.unix

    return ts
