# @Date    : 2022-08-29 18:09:33
# @Credit  : Bo Peng(bp392@cornell.edu)
# @Name    : analyze.py
# @Version : 2.0
"""
A module to carry out many low level analysis of zeus2 data
"""

import gc
import multiprocessing

from scipy.optimize import curve_fit

try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    pass

from sklearn.decomposition import FastICA

from .io import *

MAX_THREAD_NUM = multiprocessing.cpu_count()

try:  # load bias line - MCE column map table
    COL_BIAS_TB = Tb.read(pkgutil.get_data(
            __name__, "resources/column_line_map.csv").decode("utf-8"),
                          format="ascii.csv")
    COL_BIAS_MAP = {col_idx: bias_idx for (col_idx, bias_idx) in
                    zip(COL_BIAS_TB["mce_col"], COL_BIAS_TB["bias_idx"])}
except Exception as err:
    warnings.warn("Failed to load MCE column - bias line map resource " +
                  "due to %s: %s" % (type(err), err), UserWarning)
    COL_BIAS_MAP = None

if hasattr(np, "__mkl_version__"):
    warnings.warn("""
    Intel dist Python has a problem with Numpy such that operations on very
    large arrays may create a lock, which makes any future multiprocessing 
    threads wait forever, breaking the parallelization function. If you notice 
    that the program seems to frozen with parallel=True, try running it again 
    with parallel=False.""", UserWarning)


# ============================ helper functions ================================


def check_parallel():
    """
    check whether parallelization is supported on the machine, requiring starmap
    method for multiprocessing.Pool class, and MAX_THREAD_NUM > 1.

    :return: bool flag indicating if parallelization is supported
    :rtype: bool
    """

    flag = False
    if not hasattr(multiprocessing.pool.Pool, "starmap"):
        warnings.warn("Parallelization is not supported on this machine: " +
                      "multiprocessing.Pool.starmap() is not found, " +
                      "need Python >= 3.3.", UserWarning)
    elif MAX_THREAD_NUM <= 2:
        warnings.warn("Parallelization is not supported on this machine: " +
                      "MAX_THREAD_NUM <= 2.", UserWarning)
    else:
        flag = True

    return flag


def parallel_run(func, args_list):
    """
    a helper function to run in parallel

    :param func: function to run in parallel
    :param list args_list: list of args as input for the input func
    """

    gc.collect()
    num_thread = np.clip(len(args_list), a_min=2,
                         a_max=int(MAX_THREAD_NUM * 1 / 2))
    with multiprocessing.get_context("fork").Pool(
            min(MAX_THREAD_NUM, len(args_list))) as pool:
        print("running in parallel on %i threads." % num_thread)
        results = pool.starmap(func, args_list)

    return results


def transmission_smoothed_range(freq_ran, pwv, elev=60, r=1000):
    """
    Compute transmission (curve) smoothed according to the given spectral
    resolution using the kappa and eta_0 data recorded in TRANS_TB, in the range
    of frequency specified in freq_ran. The function convolve the
    transmission curve from transmission_raw_range() with a gaussian peak of
    fwhm=freq_rep/R.

    :param freq_ran: list or tuple or array, the range of frequency in unit
        GHz to compute transmission, must be within the range [400, 1610); the
        largest and the smallest values will be interpreted as the range
    :type freq_ran: list or tuple or numpy.ndarray
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :param float r: float, the spectral resolution, defining the gaussian kernel
        by fwhm=1/R*freq.mean()
    :return: (freq, trans_smoothed), arrays recording the frequency and the
        smoothed transmission computed in the given frequency range, at specified
        pwv and elevation, spectral resolution
    :rtype: list
    :raises RuntimeError: transmission data not loaded
    :raises ValueError: freq out of the range
    """

    freq_min, freq_max = np.min(freq_ran), np.max(freq_ran)
    if (freq_min < TRANS_TB["freq"].min()) or \
            (TRANS_TB["freq"].max() < freq_max):
        raise ValueError("Input frequency out of the range.")

    freq_rep = (freq_min + freq_max) / 2  # representative frequency
    freq_res = freq_rep / r  # FWHM of the resolution element
    flag_use = (freq_min - freq_res * 5 < TRANS_TB["freq"]) & \
               (TRANS_TB["freq"] < freq_max + freq_res * 5)
    freq_use, trans_use = transmission_raw_range(
            freq_ran=TRANS_TB["freq"][flag_use], pwv=pwv, elev=elev)

    gauss_kernel = gaussian(freq_use, x0=freq_use.mean(),
                            sigma=freq_res / 2 / np.sqrt(2 * np.log(2)),
                            amp=1 * np.diff(freq_use).mean(), norm=True)
    trans_smoothed_use = convolve(trans_use, gauss_kernel, mode="same")

    flag_use = (freq_min - 0.10 < freq_use) & \
               (freq_use < freq_max + 0.10)
    freq_use, trans_smoothed_use = freq_use[flag_use], \
                                   trans_smoothed_use[flag_use]

    return freq_use, trans_smoothed_use


def transmission_smoothed(freq, pwv, elev=60, r=1000):
    """
    Compute smoothed transmission (curve) by calling
    transmission_smoothed_range(), and resample at input freq.

    :param freq: int or float or array, the frequency in unit GHz to compute
        transmission, must be within the range [400, 1610); the middle value of
        input freq will be used as the representative frequency to calculate
        resolution
    :type freq: int or float or numpy.ndarray
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :param float r: float, the spectral resolution, defining the gaussian kernel
        by fwhm=1/r*freq.mean()
    :return: variable or array recording the transmission computed at the given
        frequency, pwv and elevation, and smoothed to the given spectral
        resolution, in the same shape as input freq
    :rtype: int or float or numpy.ndarray
    :raises RuntimeError: transmission data not loaded
    :raises ValueError: freq out of the range
    """

    freq_use, trans_smoothed_use = transmission_smoothed_range(
            freq_ran=freq, pwv=pwv, elev=elev, r=r)
    trans_smoothed = np.interp(freq, freq_use, trans_smoothed_use)

    return trans_smoothed


def transmission_pixel(freq, pwv, elev=60, r=1000, d_freq=0.8):
    """
    Because each pixel samples the energy in a certain range of frequency in the
    dispersed light, so the actual transmission is the smoothed curve
    (simulating the effect of grating) further convolved with a square wave
    (window) function (simulating the effect of pixel), at the central frequency.
    This function convolve the smoothed transmission curve from
    transmission_smoothed_range() with a square function characterized by the
    width d_freq.

    :param freq: int or float or array, the frequency in unit GHz to compute
        transmission, must be within the range [400, 1610); the middle value
        of input freq will be used as the representative frequency to calculate
        resolution
    :type freq: int or float or numpy.ndarray
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :param float r: float, the spectral resolution, defining the gaussian kernel
        by fwhm=1/r*freq.mean()
    :param float d_freq: float, the width of window function in unit GHz
    :return: variable or array recording the transmission computed at the given
        frequency, pwv and elevation, and smoothed to the given spectral
        resolution, in the same shape as input freq
    :rtype: int or float or numpy.ndarray
    :raises RuntimeError: transmission data not loaded
    :raises ValueError: freq out of the range
    """

    freq_min, freq_max = np.min(freq), np.max(freq)
    freq_use, trans_smoothed_use = transmission_smoothed_range(
            freq_ran=(freq_min - d_freq, freq_max + d_freq), pwv=pwv,
            elev=elev, r=r)

    sq_kernel = (abs(freq_use - freq_use.mean()) < d_freq / 2).astype(np.float)
    sq_kernel /= sq_kernel.sum()
    trans_win_use = convolve(trans_smoothed_use, sq_kernel, mode="same")

    trans_win = np.interp(freq, freq_use, trans_win_use)

    return trans_win


def bias_step_model(t, t0, period, tau, delta1, delta2, a0=0, a1=0, a2=0):
    """
    A simple model for the time series of bias step data, which combines a step
    part with an exponential decay part, both of which changes sign periodically,
    and a parabolic baseline a0 + a1*(t-t0) + a2*(t-t0)**2

    :param np.ndarray t: array object, time stamp of the bias step, ideally it
        should start from 0 or very small number
    :param float or np.double t0: float or double, the starting time stamp of the
        bias ramp
    :param float or np.double period: float or double, bias ramp period in second
    :param float tau: float, timescale of the exponential decay, or effective
        time constant
    :param float delta1: float, the size of the step change, the sign is defined as
        first chunk - the second chunk after t0
    :param float delta2: float, the scale of the exponential decay, the sign is
        defined as the scale of decay for the first chunk after t0
    :param float a0: float, zeroth order coefficient of the baseline
    :param float a1: float, first order coefficient of the baseline
    :param float a2: float, second order coefficient of the baseline
    """

    phase = (t - t0) % (period / 2)
    sign = np.choose((((t - t0) // (period / 2)) % 2).astype(int), (1, -1))

    steady = delta1 / 2 * sign
    decay = delta2 * np.exp(-phase / tau) * sign
    baseline = a0 + a1 * (t - t0) + a2 * (t - t0) ** 2

    return steady + decay


def bias_step_physics_model(t, t0, period, tau, RLR0, L, tau_bias=0, a0=0, a1=0,
                            a2=0):
    """
    A more sophisticated model of bias step based on the physical parameters of
    the TES, including the initial damped growth due to slow bias change

    :param np.ndarray t: array object, time stamp of the bias step, ideally it
        should start from 0 or very small number
    :param float or np.double t0: float or double, the starting time stamp of the
        bias ramp
    :param float or np.double period: float or double, bias ramp period in second
    :param float tau: float, the natural time constant, note that the definition is
        different from that in step_physics_model, and the effective time
        constant is tau * (1 + RLR0)/(1 + RLR0 + (1 - RLR0) * L)
    :param float RLR0: float, RL/(RL+R0), the resistance contrast
    :param float L: float, loop gain
    :param float tau_bias: float, time scale of the bias change
    :param float a0: float, zeroth order coefficient of the baseline
    :param float a1: float, first order coefficient of the baseline
    :param float a2: float, second order coefficient of the baseline
    """

    phase = (t - t0) % (period / 2)
    sign = np.choose((((t - t0) // (period / 2)) % 2).astype(int), (1, -1))

    step_eff = (1 - L) / (1 + L)
    tau_eff = tau * (1 + RLR0) / (1 + RLR0 + (1 - RLR0) * L)
    tau_r = tau_bias / tau_eff
    di_di = RLR0 * step_eff  # di_tes_di_bias
    step0 = RLR0 * (step_eff * (1 - 1 / tau_r + 1 / tau_r * np.exp(-tau_r)) +
                    1 / tau_r * (1 - np.exp(-tau_r)))

    steady = di_di / 2 * sign
    decay = (step0 - di_di) * np.exp(-(phase - tau_bias) / tau_eff) * sign * \
            (phase > tau_bias)
    increase = (RLR0 * (step_eff * (phase / tau_bias - 1 / tau_r +
                                    1 / tau_r * np.exp(-phase / tau_eff)) +
                        1 / tau_r * (1 - np.exp(-phase / tau_eff))) - di_di) * sign * \
               (phase < tau_bias)
    baseline = a0 + a1 * (t - t0) + a2 * (t - t0) ** 2

    return steady + decay + increase + baseline


def gaussian_filter_obs(obs, freq_sigma=0.3, freq_center=0,
                        edge_chunks_ncut=None, chunk_edges_ncut=None,
                        truncate=4.0):
    """
    Call scipy.ndimage.gaussian_filter1d() to do apply gaussian filtering on the
    time axis of input Obs or ObsArray object. Will first resample obs_fft to
    even and finer intervals, do smoothing, then sample back to the original
    time stamps by interpolation. Return an object the same as the original one
    except for the data_. edge_chunks_ncut is passed to
    Chop.get_flag_edge_chunks() to determine the number of edge chunks to be
    replaced by np.nan. If edge_chunks_ncut is left as None, will null the data
    with time stamp within 0.5s/(freq_sigma + freq_center) to the edge. If you
    would like not to null any edge chunks, set edge_chunks_ncut=0.

    :param obs: Obs or ObsArray object to ifft
    :type obs: Obs or ObsArray
    :param float freq_sigma: float, the standard deviation for Gaussian kernel
        to convolved with data in terms of frequency.
    :param float freq_sigma: float, the central frequency of the Gaussian kernel
    :param freq_center: center frequency of the gaussian filter
    :param edge_chunks_ncut: int or float, passed to Chop.get_flag_edge_chunks()
        as ncut to find edge chunks to null. If left None, will flag data within
        1s/freq_sigma to the edge of observation.
    :type edge_chunks_ncut: int or float
    :param chunk_edges_ncut: int or float, passed to Chop.get_flag_chunk_edges()
        as ncut to find edge chunks to null. If left None, will flag data within
        1s/freq_sigma to the edge of observation.
    :type chunk_edges_ncut: int or float
    :param float truncate: float, fft will be truncated at the frequency
        freq_sigma*truncate away from the freq_center
    :return: new object with only data_ smoothed and other attributes the same
        as input
    :rtype: Obs or ObsArray
    :raises TypeError: invalid input type
    :raises ValueError: empty ts
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    if obs.ts_.empty_flag_ or (obs.len_ == 0):
        raise ValueError("Empty ts_.")

    ts = obs.ts_
    obs_fft = fft_obs(obs=obs)
    freq_center, freq_sigma = abs(freq_center), abs(freq_sigma)
    gaussian_kern = gaussian(x=abs(obs_fft.ts_.data_), x0=freq_center,
                             sigma=freq_sigma, amp=1, norm=False)
    gaussian_kern[
        (abs(obs_fft.ts_.data_) > freq_sigma * truncate + freq_center) |
        (abs(obs_fft.ts_.data_) < - freq_center - freq_sigma * truncate)] = 0
    obs_fft.update_data(obs_fft.data_ * gaussian_kern)
    obs_ifft = ifft_obs(obs_fft)
    obs_smoothed = obs_ifft.resample_by_ts(
            ts_new=ts, method="interpolation", fill_value=0.)

    chop_freq = obs.get_chop_freq()
    if edge_chunks_ncut is None:
        edge_chunks_ncut = int(round(chop_freq / freq_sigma))
    elif not isinstance(edge_chunks_ncut, (int, float, np.integer, np.float)):
        raise TypeError("Invalid input type for edge_chunks_ncut.")
    if chunk_edges_ncut is None:
        if (edge_chunks_ncut > 0) or (freq_sigma <= 3 * chop_freq):
            chunk_edges_ncut = 0
        else:
            chunk_edges_ncut = int(round(0.5 / freq_sigma / obs.ts_.interv_))
    elif not isinstance(chunk_edges_ncut, (int, float, np.integer, np.float)):
        raise TypeError("Invalid input type for edge_chunks_ncut.")
    flag_arr = obs.chop_.get_flag_edge_chunks(ncut=edge_chunks_ncut)
    flag_arr = flag_arr | obs.chop_.get_flag_chunk_edges(chunk_edges_ncut)
    if flag_arr.sum() > obs.len_ / 2:
        warnings.warn("More than half of the data is flagged in smoothing.",
                      UserWarning)
    obs_smoothed.fill_by_flag_along_axis(flag_arr=flag_arr, axis=-1,
                                         fill_value=np.nan)

    return obs.replace(arr_in=obs_smoothed.data_)


def fft_obs(obs):
    """
    Do FFT on Obs or ObsArray object, return an object of the same type as the
    input with fft data in the last axis, and ts_ recording the frequency. The
    new object will inherit all attributes except for chop_ and obs_id_arr_

    :param obs: Obs or ObsArray object to do fft
    :type obs: Obs or ObsArray
    :return obs_fft: Obs or ObsArray object with the fft data in last axis of
        data_, and frequency in ts_
    :rtype: Obs or ObsArray
    :raises TypeError: invalid input type
    :raises ValueError: empty ts_
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    if obs.ts_.empty_flag_ or (obs.len_ == 0):
        raise ValueError("Empty ts_.")

    ts = obs.ts_
    if np.all(abs(np.diff(np.diff(obs.ts_.data_))) < obs.ts_.interv_ * 1E-3):
        interv, ts_new = ts.interv_, ts
    else:
        interv = ts.interv_ / 2
        ts_new = TimeStamps(arr_in=np.arange(ts.t_start_ - 2 * interv,
                                             ts.t_end_ + 2 * interv, interv))
    obs_interp = obs.resample_by_ts(ts_new=ts_new, method="interpolation",
                                    fill_value=0)

    data_fft = np.fft.fft(obs_interp.data_, axis=-1)
    freq_arr = np.fft.fftfreq(n=obs_interp.len_, d=interv)
    obs_fft = obs.replace(
            arr_in=data_fft, ts=freq_arr, chop=None, obs_id_arr=None)

    return obs_fft


def ifft_obs(obs_fft, t_start=None):
    """
    Do ifft on Obs or ObsArray object. The input object should have fft data in
    the last axis in data_ with frequency recorded in ts_. Again the chop_ and
    obs_id_arr_ of the returned object will remain uninitialized. By default,
    use obs.t_start_time_ as the starting time to build the new ts_, unless
    t_start is given. The returned data only contains the real part of
    numpy.fft.ifft result.

    :param obs_fft: Obs or ObsArray object to ifft
    :type obs_fft: Obs or ObsArray
    :param t_start: float or str or datetime.datetime or astropy.time.Time, the
        initial time for the time stamps in the result Obs object. If left None,
        will use t_start_time_ in obs. Allowed input type are float as zeu2 like
        time stamps which is unix format in GPS frame (the same format used by
        zeus2 and can be converted using gps_ts_to_time() and time_to_gps_ts()),
        string in iso or isot format and object
    :type t_start: float or str or datetime.datetime or astropy.time.Time
    :return: Obs or ObsArray object
    :rtype: Obs or ObsArray
    :raises TypeError: invalid input type
    :raises ValueError: empty ts
    """

    if not isinstance(obs_fft, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    if obs_fft.ts_.empty_flag_ or (obs_fft.len_ == 0):
        raise ValueError("Empty ts_.")

    ts, n = obs_fft.ts_, obs_fft.len_
    freq_interv = ts.interv_
    data_ifft = np.fft.ifft(obs_fft.data_, axis=-1).real
    ts_ifft_arr = np.arange(n, dtype=np.double) / n / freq_interv
    if t_start is None:
        t_start = obs_fft.t_start_time_
    if isinstance(t_start, (int, float, np.integer, np.double)):
        t_start = gps_ts_to_time(t_start)
    elif isinstance(t_start, str):
        try:
            t_start = Time(t_start, format="iso")
        except ValueError:
            try:
                t_start = Time(t_start, format="isot")
            except ValueError:
                raise ValueError("String format should be ISO or ISOT.")
    t_start = Time(t_start)
    ts0 = time_to_gps_ts(t_start)
    ts_ifft_arr += ts0

    return type(obs_fft)(
            arr_in=data_ifft, ts=ts_ifft_arr, chop=None, obs_id_arr=None)


def nfft_obs(obs, nfft=5., noverlap=4.):
    """
    Do fft for overlapping blocks of data in time axis, to get the dynamical
    spectrum. The return object will have data stored in the last 2 dimensions,
    the last dimension is the average time for each bin of data, and the second
    last dimension is the fft result of each block.

    :param obs: Obs or ObsArray object to do fft
    :type obs: Obs or ObsArray
    :param nfft: number of data points in each block to do fft for int input, or
        seconds of data for each block for float input
    :type nfft: int or float
    :param noverlap: number of data points that overlap between blocks, or
        seconds of overlapping data points between blacks. If the value of
        noverlap >= nfft, then nfft number of data point - 1 will be used
    :type noverlap: int or float
    :return: tuple of (Obs/ObsArray object, TimeStamp object), the first
        contains the result of fft, and the TimeStamp object stores the
        frequency information along the second last axis
    :rtype: tuple
    :raises TypeError: invalid input type of obs, nfft, noverlap
    :raises ValueError: empty ts_
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    if obs.ts_.empty_flag_ or (obs.len_ == 0):
        raise ValueError("Empty ts_.")

    ts = obs.ts_
    if np.all(abs(np.diff(np.diff(obs.ts_.data_))) < obs.ts_.interv_ * 1E-3):
        interv, ts_new, obs_interp = ts.interv_, ts, obs
    else:
        interv = ts.interv_ / 2
        ts_new = TimeStamps(arr_in=np.arange(ts.t_start_ - 2 * interv,
                                             ts.t_end_ + 2 * interv, interv))
        obs_interp = obs.resample_by_ts(ts_new=ts_new, method="interpolation",
                                        fill_value=0)

    if isinstance(nfft, (float, np.double)):
        nfft = int(nfft / interv)
    if isinstance(nfft, (int, np.integer)):
        if nfft > obs_interp.len_:
            nfft = obs_interp.len_
    else:
        raise TypeError("Invalid input type for nfft.")

    if isinstance(noverlap, (float, np.double)):
        noverlap = int(noverlap / interv)
    if isinstance(noverlap, (int, np.integer)):
        if noverlap >= nfft:
            noverlap = nfft - 1
    else:
        raise TypeError("Invalid input type for noverlap.")

    nblock = 1 + int((obs_interp.len_ - nfft) / (nfft - noverlap))
    freq_arr = np.fft.fftfreq(n=nfft, d=interv)
    ts_arr = np.empty(nblock, dtype=np.double)
    nfft_arr = np.full(obs_interp.shape_[:-1] + (nfft, nblock),
                       dtype="complex128", fill_value=np.nan)
    gc.collect()
    for i in range(nblock):
        idx_i = i * (nfft - noverlap)
        idx_e = idx_i + nfft
        ts_arr[i] = ts_new.data_[idx_i:idx_e].mean()
        nfft_arr[..., i] = np.fft.fft(obs_interp.data_[..., idx_i:idx_e],
                                      axis=-1)

    gc.collect()
    obs_nfft = obs.replace(
            arr_in=nfft_arr, ts=ts_arr, chop=None, obs_id_arr=None)
    freq_ts = TimeStamps(arr_in=freq_arr)

    return obs_nfft, freq_ts


def apply_along_axis_obs(func1d, axis, obs, *args, expand_dims=True, **kwargs):
    """
    Apply any function to the obs.data_ in obs by calling np.apply_along_axis(),
    but mind the output may have different dimension and shape. If the function
    removes the one dimension on the axis it is applied on, expand_dims can be
    set to true so that np.expand_dims() will be called to recover that dimension
    if new ndim is smaller than the original ndim. If the length in the specific
    axis doesn't match with ts, chop, obs_id_arr, or array_map (for ObsArray)
    class variable, these variables will be reset to empty.

    :param function func1d: function that operates on an 1d array, it will be
        called as func1d(arr, *args, **kwargs), passed to np.apply_along_axis()
    :param int axis: int, axis along which arr is sliced, passed to
        np.apply_along_axis()
    :param Obs or ObsArray obs: Obs or ObsArray object, the obs.data_ is
        passed to np.apply_along_axis()
    :param bool expand_dims: bool flag, in the case that new ndim is less than
        the original ndim, whether to call np.expand_dims() to recover the lost
        dimension
    :param args: arguments for func1d, passed to np.apply_along_axis()
    :param kwargs: keyword arguments for func1d, passed to np.apply_along_axis()
    :return: Obs or ObsArray object with data_ replaced by the processed array,
        and caution that ts, chop, obs_id_arr, and array_map may be reset to
        empty
    :rtype: Obs or ObsArray
    """

    arr_new = np.apply_along_axis(func1d=func1d, axis=axis, arr=obs.data_,
                                  *args, **kwargs)
    if (arr_new.ndim < obs.ndim_) and expand_dims:
        arr_new = np.expand_dims(arr_new, axis=axis)

    replace_kwargs = {"arr_in": arr_new}
    for obj, kw in zip((obs.chop_, obs.ts_, obs.obs_id_arr_),
                       ("chop", "ts", "obs_id_arr")):
        if (not obj.empty_flag_) and (obj.len_ != arr_new.shape[-1]):
            replace_kwargs[kw] = None
    if isinstance(obs, ObsArray) and (obs.array_map_.len_ != arr_new.shape[0]):
        replace_kwargs["array_map"] = None

    return obs.replace(**replace_kwargs)


def apply_along_time_obs(func1d, obs, *args, **kwargs):
    """
    Apply any function to the obs.data_ in obs by calling
    np.apply_along_axis_obs(..., axis=-1, expand_dims=True). It always try to
    preserve the dimension, and in the case the processed array have different
    length than chop, ts or obs_id_arr, the variable will be reset

    :param function func1d: function that operates on an 1d array, it will be
        called as func1d(arr, *args, **kwargs), passed to np.apply_along_axis()
    :param Obs or ObsArray obs: Obs or ObsArray object, the obs.data_ is
        passed to np.apply_along_axis()
    :param args: arguments for func1d, passed to np.apply_along_axis()
    :param kwargs: keyword arguments for func1d, passed to np.apply_along_axis()
    :return: Obs or ObsArray object with data_ replaced by the processed array,
        and caution that ts, chop, obs_id_arr, and array_map may be reset to
        empty
    :rtype: Obs or ObsArray
    """

    return apply_along_axis_obs(func1d=func1d, axis=-1, obs=obs, *args,
                                expand_dims=True, **kwargs)


def dot_prod_obs(obs1, obs2):
    """
    Perform dot product of two Obs/ObsArray objects. The returned object will be
    of the same type as obs1, and all the observation related attributes like
    obs_id and array_map will be inherited from obs1. If obs2.ndim_ == 1, then
    a sum product will be performed, the returned object shape_ will be as
    obs1.shape_[:-1], and the ts_, chop_, obs_id_arr_ will not be initialized.
    If obs2.ndim_ > 1, the result will be like numpy.dot(), and ts_, chop_,
    obs_id_arr_ will be passed from obs2 to the returned object. obs1, obs2 do
    not have to be of the same type.

    :param obs1: Obs or ObsArray object
    :type obs1: Obs or ObsArray
    :param obs2: Obs or ObsArray object
    :type obs2: Obs or ObsArray
    :return: type the same as obs1 containing dot product result
    :rtype: Obs or ObsArray
    :raises TypeError: invalid input type
    :raises ValueError: empty object
    """

    if (not isinstance(obs1, Obs)) or (not isinstance(obs2, Obs)):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    if obs1.empty_flag_ or obs2.empty_flag_:
        raise ValueError("Empty object is input.")

    dot_result = np.dot(obs1.data_, obs2.data_)
    kwargs = {"array_map": obs1.array_map_} \
        if isinstance(obs1, ObsArray) else {}
    if obs2.ndim_ == 1:
        return type(obs1)(
                arr_in=dot_result, obs_info=obs1.obs_info_, obs_id=obs1.obs_id_,
                obs_id_list=obs1.obs_id_list_,
                t_start_time=obs1.t_start_time_,
                t_end_time=obs1.t_end_time_, **kwargs)
    else:
        return type(obs1)(
                arr_in=dot_result, obs_info=obs1.obs_info_, obs_id=obs1.obs_id_,
                obs_id_list=obs1.obs_id_list_,
                chop=obs2.chop_, ts=obs2.ts_, obs_id_arr=obs2.obs_id_arr_,
                t_start_time=obs1.t_start_time_,
                t_end_time=obs1.t_end_time_, **kwargs)


def fit_obs(obs, features, **kwargs):
    """
    Least mean square fit of features to obs in the last axis by calling
    apply_along_time_obs(func1d=fit_features, ...). Features must be
    an Obs object with the same length as obs_fft.ts_. Return an object of the
    same type as the input, the same shape for all but the last axis. The
    amplitude of the fit is recorded in the last axis of the data_ in the
    returned object, and the best fit model can then be derived by calling
    dot_prod_obs(amp, features)

    :param obs: Obs or ObsArray object
    :type obs: Obs or ObsArray
    :param Obs features: Obs object, containing features, can only be 2-d with
        features in the first axis, and samples of features in the second
        axis
    :param kwargs: keyword arguments passed to nanlstsq
    :return: Obs or ObsArray object with amplitude of the fit as well as only
        obs_id related attributes. The amplitudes are in the last axis, in the
        same order of input feature.
    :raises TypeError: invalid input type
    :raises ValueError: invalid length
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid type for input obs_fft, " +
                        "expect Obs or ObsArray.")
    if not isinstance(features, Obs):
        raise TypeError("Invalid type for input features, expect Obs.")
    features = Obs(arr_in=features)
    if obs.len_ != features.len_:
        raise ValueError("Inconsistent length.")
    feature_vectors = features.data_.transpose()

    fit_features = lambda arr: nanlstsq(feature_vectors, arr, **kwargs)[0]
    amp = apply_along_time_obs(func1d=fit_features, obs=obs)

    return amp


def curve_fit_obs(f, obs, xdata=None, p0=None, bounds=(-np.inf, np.inf),
                  **kwargs):
    """
    Fitting arbitrary function to the data in obs by calling
    apply_along_time_obs(func1d=nancurve_fit, ...). The input xdata, p0 and bounds
    can be flexible as explained below.


    """

    # TODO: finish


def prep_sklearn_obs(obs):
    """
    Return an Obs/ObsArray object that is ready for sklearn functions. The
    input object will be first shrunk by taking only the time stamps with finite
    data for all pixels, then time stream of each pixel will be centered at 0
    and rescaled by std.

    :param obs: input Obs/ObsArray object
    :type obs: Obs or ObsArray
    :return: object that is ready for sklearn functions, of the same type
        as the input object
    :rtype: Obs or ObsArray
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    flattened_obs = obs.flatten()

    finite_flag = np.all(np.isfinite(flattened_obs.data_), axis=0)
    sample_obs = flattened_obs.take_by_flag_along_time(flag_arr=finite_flag)
    sample_obs -= sample_obs.proc_along_time(method="mean")
    sample_obs /= sample_obs.proc_along_time(method="std")

    return sample_obs


def sklearn_obs(obs, sklearn_solver):
    """
    Use input sklearn_solver to decompose the time stream in obs, and return an
    Obs object containing the components found by sklearn_solver

    :param obs: input Obs/ObsArray object
    :type obs: Obs or ObsArray
    :param sklearn.base.BaseEstimator sklearn_solver:
    :return: Obs object containing the sources found by sklearn_solver
    :rtype: Obs
    """

    sample_obs = prep_sklearn_obs(obs)
    sources = sklearn_solver.fit_transform(sample_obs.data_.transpose())
    source_obs = Obs(arr_in=sources.transpose(), ts=sample_obs.ts_)
    source_obs = source_obs.resample_by_ts(ts_new=obs.ts_, method="exact")

    return source_obs


def adaptive_sklearn_obs(obs, sklearn_solver, verbose=False, llim=1, ulim=.5):
    """
    Use input sklearn_solver to decompose the time stream in obs, and adapt
    n_components of sklearn_solver such that if sklearn_solver.n_iter ==
    max_iter of sklearn_solver, it indicates sklearn_solver may not converge so
    sklearn_solver is re-run with n_components -= 1; if sklearn_solver.n_iter <
    max_iter of sklearn_solver, sklearn_solver is re-run with n_components += 1,
    until the optimal n_components is found that is at the edge of convergence.

    :param obs: input Obs/ObsArray object
    :type obs: Obs or ObsArray
    :param sklearn.base.BaseEstimator sklearn_solver: must have n_iter_
        attribute
    :param bool verbose: bool flag, whether to print additional information on
        adaptive n_components setup
    :param llim: int or float, lower limit of the number of n_components, if
        float then round(number_of_input_features * llim) will be used. Must be
        within the range (1, number_of_input_features) for int input and
        (0., 1.) for float input
    :param ulim: int or float, upper limit of the number of n_components, if
        float then round(number_of_input_features * llim) will be used. Must be
        within the range (1, number_of_input_features) for int input and
        (0., 1.) for float input
    :return: Obs object containing the sources found by sklearn_solver
    :rtype: Obs
    :raises ValueError: ulim < llim, does not contain n_iter_, initial
        n_components out of range
    """

    sample_obs = prep_sklearn_obs(obs)
    n_feature = sample_obs.shape_[0]
    if isinstance(llim, (float, np.float)):
        llim = int(llim * n_feature)
    if llim <= 1:
        llim = 1
    elif llim > n_feature:
        llim = n_feature
    if isinstance(ulim, (float, np.float)):
        ulim = int(ulim * n_feature)
    if ulim <= 1:
        ulim = 1
    elif ulim > n_feature:
        ulim = n_feature
    if ulim < llim:
        raise ValueError("ulim < llim.")

    params = sklearn_solver.get_params()
    tmp_solver = type(sklearn_solver)(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="FastICA did not converge. " +
                                  "Consider increasing tolerance or " +
                                  "the maximum number of iterations.")
        tmp_solver.fit(sample_obs.data_.transpose())
    if "n_iter_" not in tmp_solver.__dict__:
        raise ValueError("The solver doesn't have n_iter_, does not support " +
                         "adaptive solving.")
    if not llim <= params["n_components"] <= ulim:
        raise ValueError("Initial value of n_components out of range.")
    if tmp_solver.n_iter_ == params["max_iter"]:
        while (tmp_solver.n_iter_ == params["max_iter"]) and \
                (llim < params["n_components"]):
            if verbose:
                print("n_iter=%i == %i for n_components=%i in range [%i, %i]." %
                      (tmp_solver.n_iter_, params["max_iter"],
                       params["n_components"], llim, ulim))
            params["n_components"] -= 1
            tmp_solver = type(sklearn_solver)(**params)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        "ignore", message="FastICA did not converge. " +
                                          "Consider increasing tolerance or " +
                                          "the maximum number of iterations.")
                tmp_solver.fit(sample_obs.data_.transpose())
        if verbose:
            print("n_iter=%i for n_components=%i, converges." %
                  (tmp_solver.n_iter_, params["n_components"]))
    else:
        while (tmp_solver.n_iter_ < params["max_iter"]) and \
                (tmp_solver.n_components < ulim):
            if verbose:
                print("n_iter=%i < %i for n_components=%i." %
                      (tmp_solver.n_iter_, params["max_iter"],
                       params["n_components"]))
            params["n_components"] += 1
            tmp_solver = type(sklearn_solver)(**params)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        "ignore", message="FastICA did not converge. " +
                                          "Consider increasing tolerance or " +
                                          "the maximum number of iterations.")
                tmp_solver.fit(sample_obs.data_.transpose())
        if verbose:
            print("n_iter=%i for n_components=%i in range [%i, %i]." %
                  (tmp_solver.n_iter_, params["n_components"], llim, ulim))
        params["n_components"] -= (tmp_solver.n_iter_ == params["max_iter"])
    if verbose:
        print("Reset n_components of the solver to %i" % params["n_components"])
    sklearn_solver.set_params(n_components=params["n_components"])
    sources = sklearn_solver.fit_transform(sample_obs.data_.transpose())

    source_obs = Obs(arr_in=sources.transpose(), ts=sample_obs.ts_)
    source_obs = source_obs.resample_by_ts(ts_new=obs.ts_, method="exact")

    return source_obs


def get_match_phase_flags(chop1, chop2, match_same_phase=False):
    """
    get the flag arrays taking the part of the 2 input chop objects that matches
    in chop phase, either the same or opposite phase

    :param chop1: Chop, the first chop
    :type chop1: Chop or Obs or ObsArray
    :param chop2: Chop, the second chop
    :type chop2: Chop or Obs or ObsArray
    :param bool match_same_phase: bool, flag of whether to match the same phase,
        if False will match the opposite phase
    :return: tuple of (flag_arr1, flag_arr2) containing the bool flag of the
        chop objects that match in the chop phase
    :rtype: tuple
    """

    if isinstance(chop1, Obs):
        chop1 = chop1.chop_
    if isinstance(chop2, Obs):
        chop2 = chop2.chop_
    chop1, chop2 = Chop(chop1), Chop(chop2)

    chunk_num1, chunk_num2 = chop1.chunk_num_, chop2.chunk_num_
    chunk_edge_idxs1 = chop1.chunk_edge_idxs_
    chunk_edge_idxs2 = chop2.chunk_edge_idxs_
    chunk_len1 = np.diff(chunk_edge_idxs1)
    chunk_len2 = np.diff(chunk_edge_idxs2)
    chunk_chop1 = chop1.chunk_proc(chunk_edge_idxs1, method="mean").data_
    chunk_chop2 = chop2.chunk_proc(chunk_edge_idxs2, method="mean").data_
    flag_arr1 = np.full(chop1.len_, fill_value=False, dtype=bool)
    flag_arr2 = np.full(chop2.len_, fill_value=False, dtype=bool)

    # search through all chunks of the one with more chunks, match phase
    idx1, idx2 = 0, 0
    while (idx1 < chunk_num1) and (idx2 < chunk_num2):
        if np.choose(match_same_phase,
                     [(chunk_chop1[idx1] != chunk_chop2[idx2]),
                      (chunk_chop1[idx1] == chunk_chop2[idx2])]):
            len1, len2 = chunk_len1[idx1], chunk_len2[idx2]
            chunk_len = min(len1, len2)
            flag_arr1[chunk_edge_idxs1[idx1]:
                      chunk_edge_idxs1[idx1] + chunk_len] = True
            flag_arr2[chunk_edge_idxs2[idx2]:
                      chunk_edge_idxs2[idx2] + chunk_len] = True
            idx1 += 1
            idx2 += 1
        else:
            if (chunk_num1 - idx1) >= (chunk_num2 - idx2):
                idx1 += 1
            else:
                idx2 += 1

    return flag_arr1, flag_arr2


def get_match_phase_obs(obs1, obs2, match_same_phase=False):
    """
    use get_match_phase_flags() to take the part of the 2 input Obs/ObsArray
    objects that matches in chop phase, either the same or opposite phase

    :param obs1: the first object
    :type obs1: Obs or ObsArray
    :param obs2: the second object
    :type obs2: Obs or ObsArray
    :param bool match_same_phase: bool, flag of whether to match the same phase,
        if False will match the opposite phase
    :return: tuple of (matched_obs1, matched_obs2) containing the part that
        match in the chop phase
    :rtype: tuple
    """

    flag1, flag2 = get_match_phase_flags(chop1=obs1, chop2=obs2,
                                         match_same_phase=match_same_phase)
    matched_obs1 = obs1.take_by_flag_along_time(flag_arr=flag1)
    matched_obs2 = obs2.take_by_flag_along_time(flag_arr=flag2)

    return matched_obs1, matched_obs2


def get_match_phase_pair(obs):
    """
    use get_match_phase_obs() to separate the input Obs/ObsArray objects into
    the on and off chop parts that can be paired

    :param obs: the input object
    :type obs: Obs or ObsArray
    :return: tuple of (obs_on, obs_off) containing the part that are on and off
        chop phase that can be paired
    :rtype: tuple
    """

    obs_on, obs_off = obs.take_by_flag_along_time(chop=True), \
                      obs.take_by_flag_along_time(chop=False)
    obs_on, obs_off = get_match_phase_obs(
            obs_on, obs_off, match_same_phase=False)

    return obs_on, obs_off


def weighted_proc_along_axis(obs, method="nanmean", weight=None, axis=-1):
    """
    Calculate the weighted mean or median for input Obs/ObsArray data along the
    given axis, return the mean or median, error and the summed weight

    :param obs: Obs or ObsArray object containing data
    :type obs: Obs or ObsArray
    :param str method: method of mean calculation, supported values are 'mean',
        'median', 'nanmean', 'nanmedian'
    :param weight: Obs or ObsArray object containing weight, should be the same
        type as obs. If left None, will treat all data point as the same weight.
    :type weight: Obs or ObsArray
    :param int axis: the axis along which the mean of obs.data_ will be
        calculated
    :return: tuple of (mean, error, summed_weight) object, the type is the same
        as the input type
    :rtype: tuple
    :raises TypeError: invalid input type
    :raises ValueError: invalid method value, inconsistent shape
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type for obs, expect Obs/ObsArray.")
    mean_obs = obs.proc_along_axis(method=method, axis=axis)
    if obs.empty_flag_:
        return mean_obs, mean_obs.copy(), mean_obs.copy()
    if method.strip().lower()[0] == "n":
        nan_policy = "omit"
        if method.strip().lower()[:6] == "nanmea":
            func = weighted_mean
        elif method.strip().lower()[:6] == "nanmed":
            func = weighted_median
        else:
            raise ValueError("Invalid value for method")
    elif method.strip().lower()[0] == "m":
        nan_policy = "propagate"
        if method.strip().lower()[:3] == "mea":
            func = weighted_mean
        elif method.strip().lower()[:3] == "med":
            func = weighted_median
        else:
            raise ValueError("Invalid value for method")
    else:
        raise ValueError("Invalid value for method")
    axis = obs.__check_axis__(axis=axis)

    data_arr = obs.data_
    if weight is None:
        wt_arr = np.ones(data_arr.shape)
    else:
        if weight.shape_ != obs.shape_:
            raise ValueError("Inconsistent shape.")
        wt_arr = weight.data_
    ni, nk = obs.shape_[:axis], obs.shape_[axis + 1:]
    mean_arr = np.empty(ni + (1,) + nk, dtype=np.double)
    err_arr = np.empty(ni + (1,) + nk, dtype=np.double)
    summed_wt_arr = np.empty(ni + (1,) + nk, dtype=wt_arr.dtype)
    for ii in np.ndindex(ni):
        for kk in np.ndindex(nk):
            mean_arr[ii + (0,) + kk], err_arr[ii + (0,) + kk], \
            summed_wt_arr[ii + (0,) + kk] = func(
                    arr=data_arr[ii + np.s_[:, ] + kk],
                    wt=wt_arr[ii + np.s_[:, ] + kk], nan_policy=nan_policy)
    wt_mean_obs = mean_obs.replace(arr_in=mean_arr)
    err_obs = mean_obs.replace(arr_in=err_arr)
    summed_wt_obs = mean_obs.replace(arr_in=summed_wt_arr)

    return wt_mean_obs, err_obs, summed_wt_obs


def get_chop_flux(obs, chunk_method="nanmedian", method="nanmean",
                  err_type="internal", weight=None, on_off=True):
    """
    Calculate the flux in chopped data. The data will first be processed in each
    chop chunk by chunk_method, unless the chunk_method is set to None or
    'none' and the data will be left as it is. Then the data will be separated
    into on-chop and off-chop part, by which the difference is the flux. The
    function supports two ways to calculate error: if err_type is 'internal',
    the difference between mean of all on and off chop data is the flux, and
    the combined error of the two parts of the data is the final error; if
    err_type is 'external', then the difference of each on-off pair will be
    taken in the first step, and then the mean and error of these differences
    is used. The method of calculating mean in this step is denoted by the
    variable method, which supports 'mean', 'nanmean', 'median', 'nanmedian'.

    :param obs: Obs or ObsArray object containing data and chop_
    :type obs: Obs or ObsArray
    :param str chunk_method: str, method parameter passed to chunk_proc() to
        chunk the data as the first step. If set to None or 'none', the data
        will skip the chunk step and the flux will be extracted from the raw
        data
    :param str method: str, the method parameter passed to
        weighted_proc_along_axis() to calculate the flux and error, suggested
        values are "nanmean" or "nanmedian"
    :param str err_type: str, allowed values are 'internal' and 'external'
    :param weight: Obs or ObsArray object containing weight, should be the same
        type as obs. If left None, will treat all data point as the same weight.
    :type weight: Obs or ObsArray
    :param bool on_off: bool flag of flux calculation using on chop - off chop,
        if False, flux is off chop - on chop
    :return: tuple of (flux, error, weight) objects of the same type as input
        obs
    :rtype: tuple
    :raises TypeError: invalid input type
    :raises ValueError: invalid method value
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type for obs, expect Obs/ObsArray.")
    obs = obs.copy()
    mean_obs = obs.proc_along_time(method="nanmean")
    if obs.empty_flag_ or obs.chop_.empty_flag_:
        raise ValueError("obs data_ or chop_ is empty.")
    if weight is None:
        weight = obs.replace(arr_in=np.ones(obs.shape_))
    weight = weight.copy()
    weight.fill_by_mask(mask=np.isnan(obs.data_), fill_value=np.nan)

    if (chunk_method is None) or chunk_method.strip().lower() == "none":
        obs_chunk_on = obs.take_by_flag_along_time(chop=True)
        obs_chunk_off = obs.take_by_flag_along_time(chop=False)
        wt_chunk_on = weight.take_by_flag_along_time(flag_arr=obs.chop_.data_)
        wt_chunk_off = weight.take_by_flag_along_time(flag_arr=~obs.chop_.data_)
    else:
        obs_chunk = obs.chunk_proc(method=chunk_method)
        obs_chunk_on = obs_chunk.take_by_flag_along_time(chop=True)
        obs_chunk_off = obs_chunk.take_by_flag_along_time(chop=False)
        wt_chunk_method = "nansum" if chunk_method.strip().lower()[:3] == "nan" \
            else "sum"
        wt_chunk = weight.chunk_proc(chunk_edge_idxs=obs.chop_.chunk_edge_idxs_,
                                     method=wt_chunk_method)
        wt_chunk_on = wt_chunk.take_by_flag_along_time(
                flag_arr=obs_chunk.chop_.data_)
        wt_chunk_off = wt_chunk.take_by_flag_along_time(
                flag_arr=~obs_chunk.chop_.data_)

    if err_type.strip().lower()[0] == "i":
        obs_chunk_on_mean, obs_chunk_on_err, obs_chunk_on_wt = \
            weighted_proc_along_axis(obs=obs_chunk_on, method=method,
                                     weight=wt_chunk_on, axis=-1)
        obs_chunk_off_mean, obs_chunk_off_err, obs_chunk_off_wt = \
            weighted_proc_along_axis(obs=obs_chunk_off, method=method,
                                     weight=wt_chunk_off, axis=-1)
        obs_flux = obs_chunk_on_mean - obs_chunk_off_mean
        obs_err = np.sqrt(obs_chunk_on_err ** 2 + obs_chunk_off_err ** 2)
        obs_wt = obs_chunk_on_wt + obs_chunk_off_wt
    elif err_type.strip().lower()[0] == "e":
        flag_arr1, flag_arr2 = get_match_phase_flags(
                chop1=obs_chunk_on.chop_, chop2=obs_chunk_off.chop_,
                match_same_phase=False)
        if (len(flag_arr1) != 0) and (len(flag_arr2) != 0):
            obs_chunk_on_match = obs_chunk_on.take_by_flag_along_time(
                    flag_arr=flag_arr1)
            obs_chunk_off_match = obs_chunk_off.take_by_flag_along_time(
                    flag_arr=flag_arr2)
            wt_chunk_on_match = wt_chunk_on.take_by_flag_along_time(
                    flag_arr=flag_arr1)
            wt_chunk_off_match = wt_chunk_off.take_by_flag_along_time(
                    flag_arr=flag_arr2)
            obs_chunk_diff = obs_chunk_on_match - obs_chunk_off_match
            wt_chunk_diff = 1 / (1 / wt_chunk_on_match + 1 / wt_chunk_off_match)
            wt_chunk_diff.fill_by_mask(mask=~np.isfinite(wt_chunk_diff.data_),
                                       fill_value=np.nan)
            obs_flux, obs_err, obs_wt = weighted_proc_along_axis(
                    obs=obs_chunk_diff, method=method, weight=wt_chunk_diff,
                    axis=-1)
        else:
            obs_flux, obs_err, obs_wt = (
                mean_obs.replace(
                        arr_in=np.full(mean_obs.shape_, fill_value=np.nan)),
                mean_obs.replace(
                        arr_in=np.full(mean_obs.shape_, fill_value=np.nan)),
                mean_obs.replace(
                        arr_in=np.full(mean_obs.shape_, fill_value=0)))
    else:
        raise ValueError("Invalid value for err_type.")
    if not on_off:
        obs_flux *= -1
    obs_flux = mean_obs.replace(arr_in=obs_flux.data_)
    obs_err = mean_obs.replace(arr_in=obs_err.data_)
    obs_wt = mean_obs.replace(arr_in=obs_wt.data_)

    return obs_flux, obs_err, obs_wt


def get_bias_obs(obs, bias_str=None):
    """
    Construct an Obs or ObsArray object of the bias value, using the RB_tes_bias
    string in .run file

    :param ArrayMap or Obs or ObsArray obs: ArrayMap or Obs or ObsArray, will
        determine the column used for bias and output type
    :param str bias_str: str, recorded in RB_tes_bias; if left None, will try to
        use RB_tes_bias in obs.obs_info_
    :return: Obs or ObsArray object corresponding to the input bias_str, if the
        input is ArrayMap,
    :rtype: Obs or ObsArray
    :raises TypeError: bias_str is None with obs not supported, Unsupported type
        of bias_str or obs.
    """

    if isinstance(obs, Obs):
        array_map = obs.to_obs_array().array_map_
    elif isinstance(obs, ObsArray):
        array_map = obs.array_map_
    elif isinstance(obs, ArrayMap):
        array_map = obs
    else:
        raise TypeError("Unsupported type of obs.")

    if bias_str is None:
        if not isinstance(obs, (Obs, ObsArray)):
            raise TypeError("Input bias_str is None with unsupported obs.")
        bias_str = obs.query_obs_info("RB_tes_bias")
        if not is_meaningful(bias_str):
            raise ValueError("'RB_tes_bias' is not found in %s.obs_info_" % obs)

    if isinstance(bias_str, str):
        tes_bias = np.fromstring(bias_str, sep="  ")
    elif isinstance(bias_str, (np.ndarray, list, tuple)):
        tes_bias = np.asarray(bias_str).flatten()
    else:
        raise TypeError("Unsupported type of bias_str.")

    bias_arr = np.full(array_map.len_, fill_value=0, dtype=float)
    col_mask = np.any(array_map.mce_col_[:, None] ==
                      np.asarray(list(COL_BIAS_MAP.keys())), axis=1)
    bias_idx = np.array([
        COL_BIAS_MAP[col] for col in array_map.mce_col_[col_mask]])
    bias_arr[col_mask] = tes_bias[bias_idx]

    bias_obs = ObsArray(bias_arr[:, None], array_map=array_map)
    if isinstance(obs, Obs):
        bias_obs = bias_obs.to_obs()

    return bias_obs


def configure_helper(obs, keyword, var=None, supersede=True):
    """
    a helper function to pick the best possible configuration parameter for any
    keyword variable, by first looking through obs.obs_info_, then
    obs.array_map_.get_conf() (if obs is ObsArray), and valid input var will
    either have the highest priority if supersede=True, or be the fallback value
    if no non-zero valid value is found in obs_info or array_map with
    supersede=False. The returned value will be the value found with the highest
    priority, or None if non-zero valid value is found in any of the places.

    :param Obs or ObsArray obs: Obs or ObsArray object, on which to search for
        obs.obs_info_ and obs.array_map_.get_conf()
    :param str or list or tuple keyword: str or list or tuple of str, the
        keyword name(s) to look for in obs.obs_info_ and
        obs.array_map_.get_conf(), with the priority ordered in the order of the
        keywords in the list
    :param var: the value of the highest priority (if supersede=True) or
        lowest priority (as a fallback value if supersede=False) in the search
    :param bool supersede: bool, flag whether a non-zero valid input value of var
        supersedes the search attempt, or is used a fallback value
    :return: the highest priority value found that is not 0 or None or "", and
        finite if it is scalar; if no such value is found, None will be returned
    :raises KeyError: keyword must be str or list(str)
    """

    if isinstance(keyword, str):
        keyword = [keyword]
    elif hasattr(keyword, "__iter__"):
        keyword = [str(kw) for kw in keyword]
    else:
        raise KeyError("Input keyword must be str or list(str).")

    conf = None
    # check whether to use var value
    if is_meaningful(var=var) and supersede:
        conf = var
    # search in obs.obs_info_
    if (not is_meaningful(conf)) and (not obs.obs_info_.empty_flag_):
        for kw in keyword:
            if (kw in obs.obs_info_.colnames_) and (not is_meaningful(conf)):
                conf_list = []
                for item in obs.obs_info_.table_[kw]:
                    if is_meaningful(item):
                        conf_list.append(item)
                if len(set(conf_list)) > 1:
                    warnings.warn("More than one values found in" +
                                  "obs_info_.table[%s], will average." % kw)
                    conf = np.mean(conf_list)
                    break
                elif len(set(conf_list)) == 1:
                    conf = conf_list[0]
                    break
    # search in obs.array_map_.get_conf()
    if (not is_meaningful(conf)) and isinstance(obs, ObsArray):
        for conf_kwargs in obs.array_map_.get_conf():
            for kw in keyword:
                if (kw in conf_kwargs) and (not is_meaningful(conf)):
                    item = conf_kwargs[kw]
                    if is_meaningful(item):
                        conf = item
                        break
    # check whether to use var value as a fallback
    if (not is_meaningful(conf)) and is_meaningful(var=var) and (not supersede):
        conf = var

    return conf


# TODO: fold obs


# ================= instrument specific reduction functions =====================


def stack_best_pixels(obs, ref_pixel=None, corr_thre=0.6, min_pix_num=10,
                      use_ref_pix_only=False):
    """
    Build a time stream model by finding the pixels well correlated with the
    reference pixel and stack their time streams. The pixels will be selected
    such that only the pixels with correlation with the ref_pixel higher
    than corr_thre, or the min_pix_num number of pixels with the highest
    correlation (whichever gives more pixels) will be stacked together with
    ref_pixel to produce the model. But if use_ref_pix_only is set to True,
    then it will only use the time stream of ref_pix. If ref_pix is left None,
    will automatically pick the pixel with the highest correlation with all
    other pixels as the reference pixel. But this process may take a long time
    because a correlation matrix of all the pixels in the input data will be
    calculated. After the well correlated pixels are found, they are stacked by
    first dividing by their nanstd in time axis, then take the average of all
    pixels at each time stamp.

    :param obs: Obs or ObsArray Object
    :type obs: Obs or ObsArray
    :param ref_pixel: tuple or list, the best pixel as reference, should be
        (spat, spec) for ObsArray input, or (row, column) for Obs input. If left
        None, will pick the pixel with the highest correlation with all other
        pixels as the best pixel.
    :type ref_pixel: tuple or list
    :param float corr_thre: float, threshold of correlation to select pixels to
        stack, if there are more than min_pix_num number of pixels with
        correlation higher than corr_thre.
    :param int min_pix_num: int, minimum number of the pixels to select
    :param bool use_ref_pix_only: bool, whether to use only the time stream
        of best_pix to build snake model
    :return: Obs object of shape (1, obs.len_)
    :rtype: Obs
    :raises TypeError: invalid input type
    """

    if not isinstance(obs, Obs):
        raise TypeError("Invalid input type, expect Obs or ObsArray.")
    obs_flattened = obs.flatten().copy()
    obs_flattened = obs_flattened.take_by_flag_along_axis(
            obs_flattened.proc_along_time("num_is_finite").data_ > 0, axis=0)
    obs_flattened = obs_flattened.take_by_flag_along_axis(
            ~obs_flattened.proc_along_time("num_not_is_finite").
            get_nanmad_flag(5, axis=0), axis=0)
    flattened_data = obs_flattened.data_
    masked_data = np.ma.masked_invalid(flattened_data)
    if ref_pixel is None:  # find the best pixel
        corr_mat = np.ma.corrcoef(masked_data)
        arg_best_pix = corr_mat.sum(axis=0).argmax()
        ref_pix_data = flattened_data[arg_best_pix]
    elif isinstance(obs, ObsArray):
        ref_pix_data = obs.take_where(spat_spec=ref_pixel).data_
    else:
        ref_pix_data = obs.data_[ref_pixel]

    if use_ref_pix_only:
        best_pix_data = ref_pix_data[None, ...]
    else:
        ref_pix_data = np.ma.masked_invalid(ref_pix_data)
        corr_best_pix = lambda arr: np.ma.corrcoef(arr, ref_pix_data)[0, 1]
        corr_arr = abs(np.ma.apply_along_axis(
                corr_best_pix, axis=-1, arr=masked_data))
        pixel_idx_list = np.flatnonzero(corr_arr > corr_thre)
        if len(pixel_idx_list) < min_pix_num + 1:
            corr_arr.fill_value = 0
            pixel_idx_list = corr_arr.filled().argsort()[-min_pix_num - 1:]
        best_pix_data = flattened_data[np.array(pixel_idx_list)]

    best_pix_obs = Obs(arr_in=best_pix_data,
                       chop=obs.chop_, ts=obs.ts_, obs_info=obs.obs_info_,
                       obs_id=obs.obs_id_, obs_id_list=obs.obs_id_list_,
                       obs_id_arr=obs.obs_id_arr_,
                       t_start_time=obs.t_start_time_,
                       t_end_time=obs.t_end_time_)
    chunk_var = best_pix_obs.chunk_proc("nanstd") ** 2
    chunk_var.fill_by_mask(chunk_var.data_ == 0, fill_value=np.nan)
    weight = (1 / chunk_var).proc_along_time("nansum")
    stacked_obs = (best_pix_obs * weight).proc_along_axis(method="mean", axis=0)

    return stacked_obs


def get_transmission_raw_obs_array(array_map, pwv, elev=60, **kwargs):
    """
    get an ObsArray object corresponding to the raw sky transmission of the
    array, the array_map_ of the output obs_array object will be populated by
    the transmission_raw_range() output with a new array_map corresponding to
    the frequency at the output data points. This function is primarily written
    for plotting raw transmission curve

    :param ArrayMap array_map: ArrayMap object, must have wavelength initialized
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :param dict kwargs: keyword arguments passed to ArrayMap.spec_calculator(),
        please make sure grat_idx in the original array_map.get_conf() result or
        in the input kwargs
    :return: ObsArray object of transmission
    :rtype: ObsArray
    :raises RuntimeError: array_map wavelength not initialized
    """

    if not array_map.wl_flag_:
        raise RuntimeError("array_map is not initialized with wavelength.")

    wl_edge = np.concatenate((array_map.array_wl_ - array_map.array_d_wl_ / 2,
                              array_map.array_wl_ + array_map.array_d_wl_ / 2))
    freq_arr, trans_arr = transmission_raw_range(
            wl_to_freq(wl_edge), pwv=pwv, elev=elev)
    wl_arr = freq_to_wl(freq_arr)
    spat_arr = np.unique(array_map.array_spat_)
    spat_arr, wl_arr, trans_arr = np.broadcast_arrays(
            spat_arr, wl_arr[:, None], trans_arr[:, None])
    spat, wl, trans = spat_arr.flatten(), wl_arr.flatten(), trans_arr.flatten()
    spec = array_map.spec_calculator(wl=wl, spat=spat, **kwargs)
    use_mask = np.isfinite(spec)  # using only valid spec
    spat, spec, trans, wl = spat[use_mask], spec[use_mask], trans[use_mask], \
                            wl[use_mask]

    array_map_new = ArrayMap(np.array([spat, spec, spat, spec]).transpose())
    array_map_new.conf_kwargs_ = array_map.conf_kwargs_
    if array_map.band_flag_:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=
            "The array map is incompatible with input band.")
            warnings.filterwarnings("ignore", message=
            "Setting band using array configuration.")
            array_map_new.set_band(array_map.band_)
    array_map_new.wl_flag_ = True
    array_map_new.array_wl_ = wl
    array_map_new.array_d_wl_ = 0.1 / wl_to_freq(wl) * wl

    trans_obs_array = ObsArray(arr_in=trans[:, None], array_map=array_map_new)

    return trans_obs_array


def get_transmission_obs_array(array_map, pwv, elev=60):
    """
    get an ObsArray object corresponding to the sky transmission of each pixel
    using transmission_pixel()

    :param ArrayMap array_map: ArrayMap object, must have wavelength initialized
    :param float pwv: float, the pwv in unit mm to compute transmission
    :param float elev: float, the elevation in unit degree to compute
        transmission
    :return: ObsArray object of transmission
    :rtype: ObsArray
    :raises RuntimeError: array_map wavelength not initialized
    """

    if not array_map.wl_flag_:
        raise RuntimeError("array_map is not initialized with wavelength.")

    wl, d_wl = array_map.array_wl_, array_map.array_d_wl_
    freq = wl_to_freq(array_map.array_wl_)
    r = np.nanmedian(wl / d_wl)
    d_freq = np.nanmedian(d_wl / wl * freq)
    trans = transmission_pixel(freq=freq, pwv=pwv, elev=elev, r=r, d_freq=d_freq)

    trans_obs_array = ObsArray(arr_in=trans[:, None], array_map=array_map)

    return trans_obs_array


def auto_flag_ts(obs, is_iv_curve=False, is_bias_step=False, is_flat=False,
                 mad_thre=7, std_thre_flat=2):
    """
    flag time series in obs in a standard way: first remove known glitch feature
    like weird data point at 0 and -0.0625; then flag by MAD threshold, the
    on-chop and off-chop data will be flagged separately if chop_ exists, to
    avoid flagging actual signal, in the case chop_ doesn't exist and is_flat=True
    the data will be flagged by STD, as MAD is prone to bimodal data; the default
    MAD or STD threshold are defined in the package variable MAD_THRE_BEAM and
    STD_THRE_FLAT

    :param obs: Obs or ObsArray object containing the time series
    :type obs: Obs or ObsArray
    :param bool is_iv_curve: bool, flag for IV curve, having higher priority than
        is_flat and is_bias_step, will flag all parts of data that is
        superconducting
    :param bool is_bias_step: bool, flag for bias step, higher priority than
        is_flat, will only flag glitch
    :param bool is_flat: bool, flag indicating this beam is flat field, which will
        use much larger mad flag threshold in the variable MAD_THRE_FLAT
    :param int or float mad_thre: int or float, MAD threshold for flagging ordinary
        time series, not applicable if is_flat=True, default 7
    :param int or float std_thre_flat: int or float, STD threshold for flagging flat
        time series, only applied if is_flat=True, default 2
    """

    obs_new = obs.copy()

    if is_iv_curve:
        iv_mask = get_iv_curve_mask(obs)
        obs_new.fill_by_mask(iv_mask, fill_value=np.nan)
    else:
        glitch_mask = (obs_new.data_ == -0.0625) | \
                      (obs_new.data_ == -112723640.0625) | \
                      (obs_new.data_ == 109850542.375) | \
                      (obs_new.data_ == 109848204.125)  # known outlier values
        blank_mask = (obs_new.data_ == 0.)  # find 0 data points
        blank_mask *= (blank_mask.sum(axis=-1, keepdims=True) <
                       0.1 * blank_mask.shape[-1])  # ignore dead pixels
        obs_new.fill_by_mask(glitch_mask | blank_mask, fill_value=np.nan)

        if not is_bias_step:
            if obs_new.chop_.empty_flag_:  # no chop data
                if is_flat:
                    outlier_mask = abs(obs_new - obs_new.proc_along_time(
                            method="nanmean")).data_ > \
                                   std_thre_flat * obs_new.proc_along_time(
                            method="nanstd").data_
                else:  # flag by MAD
                    outlier_mask = obs_new.get_double_nanmad_flag(
                            thre=mad_thre, axis=-1)
            else:  # flat on and off chop separately by MAD
                outlier_mask = np.full(obs_new.shape_, fill_value=False, dtype=bool)
                outlier_mask[..., obs_new.chop_.data_] = \
                    obs_new.take_by_flag_along_time(chop=True). \
                        get_double_nanmad_flag(thre=mad_thre, axis=-1)
                outlier_mask[..., ~obs_new.chop_.data_] = \
                    obs_new.take_by_flag_along_time(chop=False). \
                        get_double_nanmad_flag(thre=mad_thre, axis=-1)
            obs_new.fill_by_mask(mask=outlier_mask, fill_value=np.nan)

    return obs_new


def auto_flag_pix_by_ts(obs, thre_flag=5E7):
    """
    automatically flag pixel by time series if there is any datapoint above the
    given threshold, return a list of [spat, spec] or [row, col] of pixels to
    flag, depending on the input type of obs

    :param obs: Obs or ObsArray object containing the time series
    :param float thre_flag: float, absolute value threshold of the time series to
        flag a pixel, default 5E7
    :return: list, [[spat, spec], ...] if obs is ObsArray type,
        [[row, col], ...] if obs is Obs
    :rtype: list
    """

    obs_array = ObsArray(obs)
    array_map = obs_array.array_map_

    return array_map.take_by_flag(np.any(abs(
            obs_array.data_) > thre_flag, axis=-1)).array_idxs_.tolist()


def auto_flag_pix_by_flux(obs_flux, obs_err, pix_flag_list=None, is_flat=False,
                          is_bias_step=False, snr_thre=50, mad_thre_err=10):
    """
    automatically flag pixel by the flux and error of the beam, return a list of
    [spat, spec] or [row, col] of pixels to flag, depending on the input type

    :param obs_flux: Obs or ObsArray, object containing the flux
    :type obs_flux: Obs or ObsArray
    :param obs_err: Obs or ObsArray, object containing the error
    :type obs_err: Obs or ObsArray
    :param list or None pix_flag_list: list, [[spat, spec], ...] of the pixel not
        to consider in the auto flagging procedure, which increases the
        sensitivity to bad pixels
    :param bool is_bias_step: bool, flag whether the input is is_bias_step,
        overriding is_flat
    :param bool is_flat: bool, flag whether the input is a flat field, which follows
        some flagging criteria
    :param int or float snr_thre: int or float, SNR threshold of flat for a pixel
        not to be flagged, only matters if is_flat=True, default 50
    :param int or float mad_thre_err: int or float, MAD threshold for flagging
        pixel based on error, not applicable if is_flat=True, default 10
    :return: list of pixels to flag
    :rtype: list
    """

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    obs_flux_array, obs_err_array = ObsArray(obs_flux), ObsArray(obs_err)
    array_map = obs_flux_array.array_map_
    pix_flag = ~np.isfinite(obs_flux_array.data_) | \
               ~np.isfinite(obs_err_array.data_)
    if is_flat and not is_bias_step:
        pix_flag = pix_flag | \
                   (abs(obs_flux_array.data_) < 10) | \
                   (abs(obs_flux_array.data_) < snr_thre * obs_err_array.data_)
    else:
        pix_flag = pix_flag | obs_err_array.get_nanmad_flag(
                thre=mad_thre_err, axis=None)
    if is_bias_step:
        pix_flag = pix_flag | (obs_flux_array.data_ == 0) | \
                   (obs_err_array.data_ == 0)
        ramp_step_size = obs_err_array.query_obs_info("RB_cc_ramp_step_size")
        if is_meaningful(ramp_step_size):
            pix_flag = pix_flag | (abs(obs_flux_array.data_) >
                                   10 * abs(ramp_step_size))
    if pix_flag.ndim > 1:
        pix_flag = np.any(
                pix_flag, axis=tuple(np.arange(pix_flag.ndim, dtype=int)[1:]))
    pix_flag_list = pix_flag_list.copy() + array_map.take_by_flag(
            pix_flag).array_idxs_.tolist()

    if not is_flat:  # iterative flagging
        obs_err_excl = obs_err_array.exclude_where(spat_spec_list=pix_flag_list)
        pix_flag_excl = obs_err_excl.get_nanmad_flag(
                thre=mad_thre_err, axis=None)
        pix_flag_list += obs_err_excl.array_map_.take_by_flag(
                flag_arr=pix_flag_excl).array_idxs_.tolist()

    return np.unique(pix_flag_list, axis=0).tolist()
