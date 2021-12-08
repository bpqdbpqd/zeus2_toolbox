"""
Functions for the pipeline reduction. requirements: BeautifulSoup, numpy >= 1.13

Note: intel dist Python has a problem with Numpy such that operations on very
large arrays may create a lock, which make any future multiprocessing threads
wait forever, breaking the parallelization function. If you notice the thread
seems to frozen with parallel=True, try running it against with parallel=False
"""

import os, multiprocessing, inspect
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .view import *
from sklearn.decomposition import FastICA, PCA, FactorAnalysis, \
    DictionaryLearning, SparsePCA, MiniBatchSparsePCA, \
    MiniBatchDictionaryLearning

# define many default values used for the pipeline reduction
MATCH_SAME_PHASE = False  # default phase matching flag for stacking beam pairs,
# will match chop chunks of the opposite chop phase
STACK_FACTOR = 1  # default factor for the second beam in stacking beam pairs

MAD_THRE_BEAM = 7  # default MAD threshold for flagging ordinary observation
STD_THRE_FLAT = 2  # default MAD threshold for flagging skychop/flat
THRE_FLAG = 1E7  # absolute value threshold of the time series to flag a pixel
SNR_THRE = 50  # default SNR threshold to flag pixel for flat field
MAD_THRE_BEAM_ERR = 10  # default MAD threshold for flagging pixel based on error

CORR_THRE = 0.6  # default correlation threshold used in building snake model
MIN_PIX_NUM = 10  # default minimum number of pixels used to build snake model
FREQ_SIGMA = 0.3  # default gaussian peak width used in building/smoothing snake
EDGE_CHUNKS_NCUT = None  # default number of edge chunks to throw away
CHUNK_EDGES_NCUT = None  # default number/fraction of chunk edge data to abandon

FINITE_THRE = 0.95  # default threshold of finite data fraction for a pixel to be
# used in ICA
N_COMPONENTS_INIT = 3  # default initial n_components for FastICA per column
N_COMPONENTS_MAX = 5  # default max n_components for adaptive ICA
MAX_ITER = 500  # default max_iter for FastICA
RANDOM_STATE = 42  # default random state for FastICA
VERBOSE = False  # default flag for printing ICA status

CHUNK_METHOD = "nanmedian"  # default chunk method for chop flux
METHOD = "nanmean"  # default method for averaging chunks for chop flux

ORIENTATION = "horizontal"  # default orientation for making figure

MAX_THREAD_NUM = multiprocessing.cpu_count() - 1

NOD_PHASE = -1  # -1 means source is in on chop when beam is right, otherwise 1
NOD_COLNAME = "beam_is_R"  # column name in obs_info recording nodding phase

ZPOLD_SHAPE = (3, 3)  # default zpold shape, (az_len, elev_len) or (x_len, y_len)
ZPOLDBIG_SHAPE = (5, 5)  # default zpoldbig raster shape, (az_len, elev_len)
RASTER_THRE = 2  # default SNR requirement for not flagging a pixel

warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in less")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")


def custom_formatwarning(message, category, *args, **kwargs):
    # ignore everything except the message
    return "%s: %s\n" % (category.__name__, message)


warnings.formatwarning = custom_formatwarning


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
    elif MAX_THREAD_NUM <= 1:
        warnings.warn("Parallelization is not supported on this machine: " +
                      "MAX_THREAD_NUM <= 1.", UserWarning)
    else:
        flag = True

    return flag


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

    :param obs: Obs or ObsArray object to to ifft
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
                             sigma=freq_sigma, norm=False)
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


def fit_obs(obs, features):
    """
    Least mean square fit of features to obs in the last axis. Features must be
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

    fit_features = lambda arr: nanlstsq(feature_vectors, arr, rcond=-1,
                                        fill_value=np.nan)[0]
    amp_arr = np.apply_along_axis(fit_features, axis=-1, arr=obs.data_)
    kwargs = {"array_map": obs.array_map_} if isinstance(obs, ObsArray) else {}
    amp = type(obs)(arr_in=amp_arr, obs_info=obs.obs_info_,
                    obs_id=obs.obs_id_, obs_id_list=obs.obs_id_list_,
                    t_start_time=obs.t_start_time_,
                    t_end_time=obs.t_end_time_, **kwargs)

    return amp


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


def stack_best_pixels(obs, ref_pixel=None, corr_thre=0.6, min_pix_num=10,
                      use_ref_pix_only=False):
    """
    Build a time stream model by finding the pixels well correlated with the
    reference pixel and stack their time streams. The pixels will be selected
    such that only the pixels with correlation with the ref_pixel higher
    than corr_thre, or the min_pix_num number of pixels with the highest
    correlation (which ever gives more pixels) will be stacked together with
    ref_pixel to produce the model. But if use_ref_pix_only is set to True,
    then it will only use the time stream of ref_pix. If ref_pix is left None,
    will automatically pick the pixel with the highest correlation with all
    other pixels as the reference pixel. But this process may take a long time
    because a correlation matrix of all the pixels in the input data will be
    calculated. After the well correlated pixels are found, they are stacked by
    first dividing by their nanstd along time, then take the average of all
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
        chop phase that an be paired
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
    :param weight: Obs or ObsArray object containing weight, should of the same
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
    :param weight: Obs or ObsArray object containing weight, should of the same
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


def auto_flag_ts(obs, is_flat=False):
    """
    flag time series in obs in a standard way: first remove known glitch feature
    like weird data point at 0 and -0.0625; then flag by MAD threshold, the
    on-chop and off-chop data will be flagged separately if chop_ exists, to
    avoid flagging actual signal, in the case chop_ doesn't exist and is_flat=True
    the data will be flagged by STD, as MAD is prone to bimodal data; the default
    MAD or STD threshold are defined in the package variable MAD_THRE_BEAM and
    STD_THRE_FLAT

    :param obs: Obs or ObsArray object containing the time series
    :type obs: Union[Obs, ObsArray]
    :param bool is_flat: bool, flag indicating this beam is flat field, which will
        use much larger mad flag threshold in the variable MAD_THRE_FLAT
    """

    obs_new = obs.copy()
    glitch_mask = (obs_new.data_ == -0.0625) | \
                  (obs_new.data_ == -112723640.0625) | \
                  (obs_new.data_ == 109850542.375) | \
                  (obs_new.data_ == 109848204.125)  # known outlier values
    blank_mask = (obs_new.data_ == 0.)  # find 0 data points
    blank_mask *= (blank_mask.sum(axis=-1, keepdims=True) <
                   0.1 * blank_mask.shape[-1])  # ignore dead pixels
    obs_new.fill_by_mask(glitch_mask | blank_mask, fill_value=np.nan)

    if obs_new.chop_.empty_flag_:  # no chop data
        if is_flat:
            outlier_mask = \
                abs(obs_new - obs_new.proc_along_time(method="nanmean")).data_ > \
                STD_THRE_FLAT * obs_new.proc_along_time(method="nanstd").data_
        else:  # flag by MAD
            outlier_mask = obs_new.get_double_nanmad_flag(thre=MAD_THRE_BEAM, axis=-1)
    else:  # flat on and off chop separately by MAD
        outlier_mask = np.full(obs_new.shape_, fill_value=False, dtype=bool)
        outlier_mask[..., obs_new.chop_.data_] = \
            obs_new.take_by_flag_along_time(chop=True).get_double_nanmad_flag(
                    thre=MAD_THRE_BEAM, axis=-1)
        outlier_mask[..., ~obs_new.chop_.data_] = \
            obs_new.take_by_flag_along_time(chop=False).get_double_nanmad_flag(
                    thre=MAD_THRE_BEAM, axis=-1)
    obs_new.fill_by_mask(mask=outlier_mask, fill_value=np.nan)

    return obs_new


def auto_flag_pix_by_ts(obs, thre_flag=THRE_FLAG):
    """
    automatically flag pixel by time series if there is any datapoint above the
    given threshold, return a list of [spat, spec] or [row, col] of pixels to
    flag, depending on the input type of obs

    :param obs: Obs or ObsArray object containing the time series
    :param float thre_flag: float, absolute value threshold of the time series to
        flag a pixel, by default use THRE_FLAG
    :return: list, [[spat, spec], ...] if obs is ObsArray type,
        [[row, col], ...] if obs is Obs
    :rtype: list
    """

    obs_array = ObsArray(obs)
    array_map = obs_array.array_map_

    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="invalid value encountered in greater")
        return array_map.take_by_flag(np.any(
                abs(obs_array.data_) > thre_flag, axis=-1)). \
            array_idxs_.tolist()


def auto_flag_pix_by_flux(obs_flux, obs_err, pix_flag_list=[], is_flat=False,
                          snr_thre=SNR_THRE):
    """
    automatically flag pixel by the flux and error of the beam, return a list of
    [spat, spec] or [row, col] of pixels to flag, depending on the input type

    :param obs_flux: Obs or ObsArray, object containing the flux
    :type obs_flux: Union[Obs, ObsArray]
    :param obs_err: Obs or ObsArray, object containing the error
    :type obs_err: Union[Obs, ObsArray]
    :param list pix_flag_list: list, [[spat, spec], ...] of the pixel not to
        consider in the auto flagging procedure, which increases the sensitivity
        to bad pixels
    :param bool is_flat: bool, flag whether the input is a flat field, which follows
        some flagging criteria
    :param int snr_thre: int, SNR threshold of flat for a pixel not to be flagged,
        only matters if is_flat=True
    :return: list of pixels to flag
    :rtype: list
    """

    obs_flux_array, obs_err_array = ObsArray(obs_flux), ObsArray(obs_err)
    array_map = obs_flux_array.array_map_
    if is_flat:
        pix_flag = ~np.isfinite(obs_flux_array.data_) | \
                   ~np.isfinite(obs_err_array.data_) | \
                   (abs(obs_flux_array.data_) < 10) | \
                   (abs(obs_flux_array.data_) < snr_thre * obs_err_array.data_)
    else:
        pix_flag = obs_err_array.get_nanmad_flag(thre=MAD_THRE_BEAM_ERR, axis=None)
    if pix_flag.ndim > 1:
        pix_flag = np.any(
                pix_flag, axis=tuple(np.arange(pix_flag.ndim, dtype=int)[1:]))
    pix_flag_list = pix_flag_list.copy() + array_map.take_by_flag(
            pix_flag).array_idxs_.tolist()

    if not is_flat:  # iterative flagging
        obs_err_excl = obs_err_array.exclude_where(spat_spec_list=pix_flag_list)
        pix_flag_excl = obs_err_excl.get_nanmad_flag(
                thre=MAD_THRE_BEAM_ERR, axis=None)
        pix_flag_list += obs_err_excl.array_map_.take_by_flag(
                flag_arr=pix_flag_excl).array_idxs_.tolist()

    return np.unique(pix_flag_list, axis=0).tolist()


# ================= intermediate level reduction functions =====================


def desnake_beam(obs, ref_pix=None, pix_flag_list=[], corr_thre=CORR_THRE,
                 min_pix_num=MIN_PIX_NUM, freq_sigma=FREQ_SIGMA,
                 edge_chunks_ncut=EDGE_CHUNKS_NCUT,
                 chunk_edges_ncut=CHUNK_EDGES_NCUT):
    """
    build a snake model for the input time series data in obs, fit and subtract
    the snake model for all the pixels; return the desnaked data, snake model
    for each pixel, amplitude of the snake model and the snake model

    :param obs: Obs or ObsArray object containing the time series
    :type obs: Union[Obs, ObsArray]
    :param list ref_pix: list of 2, [spat, spec] for ObsArray input or
        [row, col] for Obs input, the reference pixel to use in desnaking; will
        automatically determine the best correlated pixel if left None
    :param list pix_flag_list: list, [[spat, spec], ...] or [[row, col], ...] of
        the pixels not to include in building the snake, passed to
        stack_best_pixels()
    :param float corr_thre: float, the minimum correlation to the reference pixel
        requirement to be used for building the snake model, passed to
        stack_best_pixels() ; use the value of CORR_THRE by default
    :param int min_pix_num: int, the minimum number of pixels to be used for
        building the snake model, passed to stack_best_pixels(); use the value
        of MIN_PIX_NUM by default
    :param float freq_sigma: float, the standard deviation for Gaussian kernel
        to smooth the stacked time series in unit of frequency, passed to
        gaussian_filter_obs() ; use the value of FREQ_SIGMA by default
    :param int edge_chunks_ncut: int, number of edge chunks to throw away,
        passed to gaussian_filter_obs(); use the value of EDGE_CHUNKS_NCUT by
        default
    :param chunk_edges_ncut: int or float, number or fraction of chunk edge
        data points to throw away, passed to gaussian_filter_obs(); use the value
        of CHUNK_EDGES_NCUT by default
    :rtype chunk_edges_ncut: Union[int, float]
    :return: list, (desnaked_obs, obs_snake, amp_snake, snake_model)
    :rtype: list
    """

    if ref_pix in pix_flag_list:
        warnings.warn("Reference pixel is in pix_flag_list, will ignore ref_pix")
        ref_pix = None
    stacked_best_pixels_obs = stack_best_pixels(
            ObsArray(obs).exclude_where(spat_spec_list=pix_flag_list),
            ref_pixel=ref_pix, corr_thre=corr_thre, min_pix_num=min_pix_num)
    snake_model = gaussian_filter_obs(
            stacked_best_pixels_obs, freq_sigma=freq_sigma,
            edge_chunks_ncut=edge_chunks_ncut,
            chunk_edges_ncut=chunk_edges_ncut)
    if np.isfinite(snake_model.data_).sum() == 0:
        warnings.warn("Data is too short for desnaking!", UserWarning)

    snake_model.expand(snake_model * 0 + 1)  # fit snake model to each pixel
    amp_snake = fit_obs(obs, features=snake_model)
    obs_snake = dot_prod_obs(amp_snake, snake_model)

    return obs - obs_snake, obs_snake, amp_snake, snake_model


def ica_treat_beam(obs, spat_excl=None, pix_flag_list=[], verbose=VERBOSE,
                   finite_thre=FINITE_THRE, n_components_init=N_COMPONENTS_INIT,
                   n_components_max=N_COMPONENTS_MAX, max_iter=MAX_ITER,
                   random_state=RANDOM_STATE):
    """
    build noise model by running FastICA decomposition on each MCE column, then
    fit and subtract noise from the input data; return the data with noise
    subtracted, noise model for each pixel, noise feature amplitude for each
    pixel and noise features

    :param obs: Obs or ObsArray, object containing the time series
    :type obs: Union[Obs, ObsArray]
    :param list spat_excl: list, the spatial position range excluded in building
        noise model using ICA, e.g. spat_excl=[0, 2] means pixels at spat=0,1,2
        will not be used; will use all pixels if let None
    :param list pix_flag_list: list, [[spat, spec], ...] of pixels to exclude
        from ICA
    :param bool verbose: bool, flag whether to print status of FastICA
    :param float finite_thre: float, fraction of finite data required for a pixel to
        be used in ICA; use FINITE_THRE by default
    :param int n_components_init: int, initial value for n_components in
        FastICA(); use N_COMPONENTS_INIT by default
    :param int n_components_max: int, max value for n_components, passed to
        adaptive_sklearn_obs() as ulim; use N_COMPONENTS_MAX by default
    :param int max_iter: int, max iteration as max_iter in FastICA(); use
        MAX_ITER by default
    :param random_state: int or float, use RANDOM_STATE by default
    :return: (ica_noise_sub_beam, ica_noise_beam, amp_ica, ica_noise_model)
    :rtype: list
    """

    obs_array, obs_sources = ObsArray(obs), Obs()
    array_map = obs_array.array_map_
    pix_excl_list = pix_flag_list + array_map.take_by_flag(
            ((obs_array.proc_along_time("num_is_finite").data_ /
              np.nanmax(obs_array.proc_along_time("num_is_finite").data_)) <
             finite_thre) |
            ((obs_array.data_ == 0).sum(axis=-1, keepdims=True) /
             obs_array.len_ > 0.5)).array_idxs_.tolist()
    obs_flattened = obs_array.flatten().exclude_where(
            spat_spec_list=pix_excl_list, spat_ran=spat_excl, logic="or")

    # run ICA
    for col in range(array_map.mce_col_llim_, array_map.mce_col_ulim_ + 1):
        obs_use = obs_array.exclude_where(
                spat_spec_list=pix_excl_list, spat_ran=spat_excl, logic="or"). \
            take_where(col=col).flatten()
        if obs_use.shape_[0] > 0:
            ica = FastICA(
                    n_components=min(n_components_init, obs_use.shape_[0]),
                    whiten=True, fun='exp', max_iter=max_iter,
                    random_state=random_state)
            obs_use = gaussian_filter_obs(
                    obs_use, freq_sigma=15, chunk_edges_ncut=0,
                    edge_chunks_ncut=0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=
                "FastICA did not converge. Consider increasing tolerance or" +
                "the maximum number of iterations.")
                obs_sources.expand(adaptive_sklearn_obs(
                        obs_use, ica, verbose=verbose, llim=1,
                        ulim=min(n_components_max, obs_use.shape_[0])))
            if verbose:
                print("MCE column = %i, iter num = %i, input pix num = %i" %
                      (col, ica.n_iter_, obs_use.shape_[0]))

    # fit to the original data
    obs_sources.expand(obs_sources.replace(
            arr_in=np.ones((1, obs.len_)), ts=obs.ts_))
    obs_to_fit = obs_sources.copy()
    obs_chop = obs_sources.replace(arr_in=obs.chop_.data_[None, ...])
    obs_to_fit.expand(obs_chop)
    ica_amp = fit_obs(obs, obs_to_fit). \
        take_by_idx_along_axis(range(obs_to_fit.shape_[0] - 1), axis=-1)
    ics_noise_beam = dot_prod_obs(ica_amp, obs_sources)

    return obs - ics_noise_beam, ics_noise_beam, ica_amp, obs_sources


def plot_beam_ts(obs, title=None, pix_flag_list=[], reg_interest=None,
                 plot_show=False, plot_save=False, write_header=None,
                 orientation=ORIENTATION):
    """
    plot time series for the pipeline reduction

    :param obs: Obs or ObsArray or list or tuple or dict, can be the object
        containing the data to plot, or list/tuple of objects, or dict in the
        form of {key: obs} or {key: (obs, kwargs)} or {key: (obs, obs_yerr)} or
        {key: (obs, obs_yerr, kwargs)} or {key: [obs, kwargs]}, in which case
        the dict key will be the label in legend, obs and obs_yerr is Obs or
        ObsArray objects, and kwargs is passed to FigArray.scatter() if the dict
        iterm is tuple or FigArray.plot() if it's list, the items in the
        tuple/list determined based on type, and if obs_yerr is present,
        FigArray.errorbar() will also be called with kwargs
    :type obs: Union[Obs, ObsArray, list, tuple, dict]
    :param str title: str, title of the figure, will use the first available
        obs_id if left None
    :param list pix_flag_list: list, [[spat, spec], ...] or [[row, col], ...] of
        the flagged pixels, shown in grey shade
    :param dict reg_interest: dict, indicating the region of array to plot,
        passed to ArrayMap.take_where(); will plot all the input pixels if
        left None
    :param bool plot_show: bool, flag whether to show the figure with plt.show()
    :param bool plot_save: bool, flag whether to save the figure
    :param str write_header: str, path to the file header to write the figure to,
        the figure will be saved as {write_header}.png, only matters if
        plot_save=True; will use the first available obs_id if left None
    :param str orientation: str, the orientation of the figure, passed to
        FigArray.init_with_array_map
    :return: FigArray, object of the figure
    :rtype: FigArray
    """

    if isinstance(obs, (Obs, ObsArray, np.ndarray)):
        obs0 = obs
    elif isinstance(obs, dict):
        obs0 = list(obs.values())[0]
        if isinstance(obs0, (list, tuple)):
            obs0 = obs0[0]
    else:
        obs0 = obs[0]
    array_map = ObsArray(obs0).array_map_
    if title is None:
        title = obs0.obs_id_
    if write_header is None:
        write_header = obs0.obs_id_
    if isinstance(obs0, (Obs, ObsArray)) and (not obs0.ts_.empty_flag_):
        obs_t_len = obs0.t_end_time_ - obs0.t_start_time_
        x_size = max((obs_t_len / units.hour).to(1).value / 2,
                     FigArray.x_size_)
    else:
        x_size = FigArray.x_size_

    fig = FigArray.init_by_array_map(array_map if reg_interest is None else
                                     array_map.take_where(**reg_interest),
                                     orientation=orientation, x_size=x_size)
    if isinstance(obs, (Obs, ObsArray, np.ndarray)):
        fig.scatter(obs)
    elif isinstance(obs, dict):
        for key in obs:
            if isinstance(obs[key], (list, tuple)):
                plot_func = fig.scatter if isinstance(obs[key], tuple) else \
                    fig.plot
                if len(obs[key]) > 1:
                    if isinstance(obs[key][1], (Obs, ObsArray)):
                        kwargs = obs[key][2] if len(obs[key]) > 2 else {}
                        plot_func(obs[key][0], **kwargs)
                        fig.errorbar(obs[key][0], yerr=obs[key][1], label=key,
                                     **kwargs)
                    else:
                        plot_func(obs[key][0], label=key, **obs[key][1])
                else:
                    plot_func(obs[key][0], label=key)
            else:
                fig.scatter(obs[key], label=key)
        fig.legend(loc="upper left")
        if fig.twin_axs_list_ is not None:
            fig.legend(twin_axes=True, loc="lower right")
    else:
        for obs_i in obs:
            fig.scatter(obs_i)
    fig.imshow_flag(pix_flag_list=pix_flag_list)
    fig.set_labels(obs0, orientation=orientation)
    fig.set_title(title)

    if plot_save:
        fig.savefig("%s.png" % write_header)
    if plot_show:
        plt.show()

    return fig


def plot_beam_flux(obs, title=None, pix_flag_list=[], plot_show=False,
                   plot_save=False, write_header=None, orientation=ORIENTATION):
    """
    plot flux for the pipeline reduction

    :param obs: Obs or ObsArray, the object containing the data to plot
    :type obs: Union[Obs, ObsArray]
    :param str title: str, title of the figure, will use the obs_id if left
        None
    :param list pix_flag_list: list, [[spat, spec], ...] or [[row, col], ...] of
        the flagged pixels, shown in grey shade
    :param bool plot_show: bool, flag whether to show the figure with plt.show()
    :param bool plot_save: bool, flag whether to save the figure
    :param str write_header: str, path to the file header to write the figure,
        the figure will be saved as {write_header}.png, only matters if
        plot_save=True; will use the first available obs_id if left None
    :param str orientation: str, the orientation of the figure, passed to
        FigFlux.plot_flux
    :return: FigFlux, the figure object
    :rtype: FigFlux
    """

    if title is None:
        title = obs.obs_id_
    if write_header is None:
        write_header = obs.obs_id_

    fig = FigFlux.plot_flux(obs, pix_flag_list=pix_flag_list,
                            orientation=orientation)
    fig.set_labels(obs, orientation=orientation)
    fig.set_title(title)
    if plot_save:
        fig.savefig("%s.png" % write_header)
    if plot_show:
        plt.show()

    return fig


def analyze_performance(beam, write_header=None, pix_flag_list=[], plot=False,
                        plot_rms=False, plot_ts=False, reg_interest=None,
                        plot_psd=True, plot_specgram=False, plot_show=False,
                        plot_save=False):
    """
    Analyze the performance of each pixel in the beam, including the rms of each
    pixel, plotting the time series, power spectral diagram (psd) and dynamical
    spectrum

    :param beam: Obs or ObsArray, with time series data
    :type beam: Union[Obs, ObsArray]
    :param str write_header: str, full path to the title to save files/figures,
        if left None, will write to current folder with obs_id as the title
    :param list pix_flag_list: list, a list including pixels to be flagged
    :param bool plot: bool, flag whether to make figure
    :param bool plot_rms: bool, flag whether to plot rms in 2-d array layout
    :param bool plot_ts: bool, flag whether to plot time series for each pixel
    :param dict reg_interest: dict, region of interest of array passed to
        ArrayMap.take_where() for plotting
    :param bool plot_psd: bool, flag whether to plot power spectral diagram
    :param plot_specgram: bool, flag whether to plot dynamical spectrum
    :param bool plot_show: bool, flag whether to call plt.show() and show figure
    :param bool plot_save: bool, flag whether to save the figure
    :return: Obs or ObsArray, containing the average chop-wise rms of each pixel
    :rtype: Union[Obs, ObsArray]
    """

    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), beam.obs_id_)
    beam_chop_rms = beam.chunk_proc(method="nanstd")
    beam_chop_rms.fill_by_mask(
            beam.chunk_proc(method="num_is_finite").data_ <= 10)
    beam_chop_rms.fill_by_mask(
            beam_chop_rms.get_nanmad_flag(thre=MAD_THRE_BEAM, axis=-1))
    beam_chop_wt = beam.chunk_proc(method="num_is_finite")
    beam_rms = weighted_proc_along_axis(beam_chop_rms, method="nanmean",
                                        weight=beam_chop_wt, axis=-1)[0]

    if plot:
        if plot_rms:
            print("Plotting rms.")
            plt.close(plot_beam_flux(
                    beam_rms, title="%s rms" %
                                    write_header.split("/")[-1],
                    pix_flag_list=pix_flag_list, plot_show=plot_show,
                    plot_save=plot_save, write_header="%s_rms" %
                                                      write_header,
                    orientation=ORIENTATION))
        print("Plotting rms time series.")
        plot_dict = {"rms": (beam_chop_rms, {"c": "k"})}
        if plot_ts and np.prod(beam.shape_) < 3 * 10 * 10000:
            plot_dict["raw data"] = (beam, {"twin_axes": True})
        plt.close(plot_beam_ts(
                plot_dict, title="%s rms" %
                                 write_header.split("/")[-1],
                pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                plot_show=plot_show, plot_save=plot_save,
                write_header="%s_rms_ts" % write_header,
                orientation=ORIENTATION))
        if plot_ts:
            plot_dict = {"raw_data": beam}
            if ("UTC" in beam.obs_info_.table_.colnames) and \
                    ("mm PWV" in beam.obs_info_.table_.colnames):
                tb_use = beam.obs_info_.table_[
                    ~beam.obs_info_.table_.mask["UTC"]]
                tb_use.sort("UTC")
                t_arr = Time.strptime(tb_use["UTC"],
                                      format_string="%Y-%m-%dU%H:%M:%S"). \
                    to_value(format="unix")
                t_arr += tb_use["Scan duration"] / 2
                pwv_arr = tb_use["mm PWV"]
                beam_pwv = beam.replace(
                        arr_in=np.tile(pwv_arr, beam.shape_[:-1] + (1,)),
                        ts=t_arr, chop=None)
                plot_dict["PWV"] = [beam_pwv, {
                    "ls": "--", "c": "r", "marker": ".", "markersize": 3,
                    "twin_axes": True}]
            print("Plotting time series.")
            plt.close(plot_beam_ts(
                    plot_dict, title="%s time series" %
                                     write_header.split("/")[-1],
                    pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                    plot_show=plot_show, plot_save=plot_save,
                    write_header="%s_ts" % write_header,
                    orientation=ORIENTATION))
        if plot_psd:
            print("Plotting power spectral diagram.")
            fig = FigArray.plot_psd(
                    beam if reg_interest is None else
                    ObsArray(beam).take_where(**reg_interest),
                    orientation=ORIENTATION, scale="dB", lw=0.5)
            fig.imshow_flag(pix_flag_list=pix_flag_list, orientation=ORIENTATION)
            fig.set_labels(beam, orientation=ORIENTATION)
            fig.set_title("%s power spectral diagram" %
                          write_header.split("/")[-1])
            fig.set_ylabel("Spectral power [dB]")
            fig.set_xlabel("Frequency [Hz]")
            if plot_show:
                plt.show(fig)
            if plot_save:
                fig.savefig("%s_psd.png" % write_header)
            plt.close(fig)
        if plot_specgram:
            print("Plotting dynamical spectrum.")
            if not beam.ts_.empty_flag_:
                beam_t_len = beam.t_end_time_ - beam.t_start_time_
                x_size = max((beam_t_len / units.hour).to(1), FigArray.x_size_)
                nfft = min(12., (beam_t_len / units.second).to(1).value,
                           max(5., (beam_t_len / units.second).to(1).value /
                               30.))
                noverlap = max(2., min(nfft - 1., nfft -
                                       (beam_t_len / units.second).to(1).value /
                                       300.))
            else:
                x_size = FigArray.x_size_
                nfft = min(12 * 240, beam.len_,
                           max(5 * 240, int(beam.len_ / 240 / 30)))
                noverlap = max(2 * 240, min(nfft - 240,
                                            nfft - int(beam.len_ / 240 / 300)))
            fig = FigArray.plot_specgram(
                    beam if reg_interest is None else
                    ObsArray(beam).take_where(**reg_interest),
                    orientation=ORIENTATION, x_size=x_size, nfft=nfft,
                    noverlap=noverlap, scale="dB", cmap="gist_ncar")
            fig.imshow_flag(pix_flag_list=pix_flag_list, orientation=ORIENTATION)
            fig.set_labels(beam, orientation=ORIENTATION)
            fig.set_title("%s dynamical spectrum" % write_header.split("/")[-1])
            fig.set_ylabel("Frequency [Hz]")
            fig.set_xlabel("GPS time [s]")
            if plot_show:
                plt.show(fig)
            if plot_save:
                fig.savefig("%s_specgram.png" % write_header)
            plt.close(fig)

    return beam_rms


def proc_beam(beam, write_header=None, is_flat=False, pix_flag_list=[], flat_flux=1,
              flat_err=0, cross=False, do_desnake=False, ref_pix=None,
              do_smooth=False, do_ica=False, spat_excl=None, return_ts=False,
              return_pix_flag_list=False, plot=False, plot_ts=False,
              reg_interest=None, plot_flux=False, plot_show=False,
              plot_save=False, chunk_method=CHUNK_METHOD, method=METHOD):
    """
    process beam in the standard way, return chop flux, error and weight

    :param beam: Obs or ObsArray, with time series data
    :type beam: Union[Obs, ObsArray]
    :param str write_header: str, full path to the title to save files/figures,
        if left None, will write to current folder with {obs_id} as the title
    :param bool is_flat: bool, flag indicating this beam is flat field, which will
        use much larger mad flag threshold, flag pixel by SNR, and will not use
        weighted mean in calculating flux
    :param list pix_flag_list: list, a list including pixels to be flagged, will
        be combined with auto flagged pixels in making figures and in the returned
        pix_flag_list, the pixels will be flagged in the figure and the process of
        modelling noise
    :param flat_flux: Obs or ObsArray, the flat field flux to divide in computing the
        beam flux, must have the same shape and array_map; will ignore if is_flat
        is True; default is 1
    :type flat_flux: Union[Obs, ObsArray, int, float]
    :param flat_err: Obs or ObsArray, the flat field flux err used in computing the
        beam error, having the same behaviour as flat; default is 0
    :type flat_err: Union[Obs, ObsArray, int, float]
    :param bool cross: bool, flag whether the beam is a cross scan; if True, will
        process the beam to get the flux in each chop chunk pair, instead the
        whole scan
    :param bool do_desnake: bool, flag whether to perform desnaking
    :param list ref_pix: list of 2, [spat, spec] for ObsArray input or
        [row, col] for Obs input, the reference pixel to use in desnaking; will
        automatically determine the best correlated pixel if left None
    :param bool do_smooth: bool, flag whether to smooth data with a gaussian
        kernel of FREQ_SIGMA
    :param bool do_ica: bool, flag whether to perform ica
    :param list spat_excl: list, the spatial position range excluded in building
        noise model using ICA, e.g. spat_excl=[0, 2] means pixels at spat=0,1,2
        will not be used; will use all pixels if let None
    :param bool return_ts: bool, flag whether to return time series object
    :param bool return_pix_flag_list: bool, bool, flag whether to return
        pix_flag_list
    :param bool plot: bool, flag whether to make figure
    :param bool plot_ts: bool, flag whether to plot time series
    :param dict reg_interest: dict, region of interest of array passed to
        ArrayMap.take_where() for plotting
    :param bool plot_flux: bool, flag whether to plot flux
    :param bool plot_show: bool, flag whether to call plt.show() and show figure
    :param bool plot_save: bool, flag whether to save the figure
    :return: tuple (beam_flux, beam_err, beam_wt, [beam_ts], [pix_flag_list]),
        are (Obs/ObsArray recording beam flux, Obs/ObsArray recording beam flux
        error, Obs/ObsArray recording beam weight, [optional, Obs/ObsArray
        recording processed time series after flagging], [optional, list of auto
        flagged pixels])
    :rtype: tuple
    """

    pix_flag_list = pix_flag_list.copy() + auto_flag_pix_by_ts(beam)  # auto flag
    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), beam.obs_id_)
    beam_use, noise_beam = beam, type(beam)()
    plot_dict = {"raw data": beam}

    if do_desnake:  # model noise
        desnaked_beam, snake_beam, amp_snake, snake_model = desnake_beam(
                beam_use, ref_pix=ref_pix, pix_flag_list=pix_flag_list)
        beam_use = auto_flag_ts(desnaked_beam, is_flat=is_flat)
        noise_beam += snake_beam
        plot_dict["snake"] = (noise_beam, {"c": "k"})
    if do_smooth:
        smooth_beam = gaussian_filter_obs(
                beam_use, freq_sigma=FREQ_SIGMA,
                edge_chunks_ncut=EDGE_CHUNKS_NCUT,
                chunk_edges_ncut=CHUNK_EDGES_NCUT).replace(chop=None)
        beam_use = auto_flag_ts(beam_use - smooth_beam, is_flat=is_flat)
        noise_beam += smooth_beam
        plot_dict["smooth"] = (noise_beam, {"c": "y"})
    if do_ica:
        ica_treated_beam, ica_noise_beam, amp_ica, ica_noise_model = \
            ica_treat_beam(beam_use, spat_excl=spat_excl,
                           pix_flag_list=pix_flag_list)
        beam_use = auto_flag_ts(ica_treated_beam, is_flat=is_flat)
        noise_beam += ica_noise_beam
        plot_dict["ica"] = (noise_beam, {"c": "gray"})

    if cross:
        beam_chunk_flux = beam_use.chunk_proc(
                method=CHUNK_METHOD if CHUNK_METHOD is not None else "nanmean")
        beam_chunk_err = beam_use.chunk_proc(method="nanstd")
        beam_chunk_wt = beam_use.chunk_proc(method="num_is_finite")
        beam_chunk_err /= beam_chunk_wt.sqrt()
        beam_chop_flux_on, beam_chop_flux_off = \
            get_match_phase_pair(beam_chunk_flux)
        beam_chop_err_on, beam_chop_err_off = \
            get_match_phase_pair(beam_chunk_err)
        beam_chop_wt_on, beam_chop_wt_off = \
            get_match_phase_pair(beam_chunk_wt)
        beam_cross_flux = beam_chop_flux_on - beam_chop_flux_off
        beam_cross_err = (beam_chop_err_on ** 2 + beam_chop_err_off ** 2).sqrt()
        beam_cross_wt = beam_chop_wt_on + beam_chop_wt_off
        beam_cross_flux, beam_cross_err = \
            beam_cross_flux / flat_flux, abs(beam_cross_flux / flat_flux) * \
            ((beam_cross_err / beam_cross_flux) ** 2 +
             (flat_err / flat_flux) ** 2).sqrt()
    beam_flux, beam_err, beam_wt = get_chop_flux(  # compute flux and error
            beam_use, chunk_method=chunk_method, method=method,
            weight=(
                None if is_flat else
                1 / (beam_use.chunk_proc("nanstd", keep_shape=True) ** 2 /
                     beam_use.chunk_proc("num_is_finite", keep_shape=True) +
                     (beam_use.chunk_proc("nanmedian", keep_shape=True) -
                      beam_use.chunk_proc("nanmean", keep_shape=True)) ** 2)),
            err_type="external")
    pix_flag_list = auto_flag_pix_by_flux(  # auto flag
            beam_flux, beam_err, pix_flag_list=pix_flag_list, is_flat=is_flat)
    beam_flux, beam_err = beam_flux / flat_flux, abs(beam_flux / flat_flux) * \
                          ((beam_err / beam_flux) ** 2 +
                           (flat_err / flat_flux) ** 2).sqrt()

    if plot and plot_flux:
        plt.close(plot_beam_flux(
                beam_flux, title="%s flux" % write_header.split("/")[-1],
                pix_flag_list=pix_flag_list, plot_show=plot_show,
                plot_save=plot_save, write_header="%s_flux" % write_header,
                orientation=ORIENTATION))
    if plot and plot_ts:
        if cross:
            plot_dict["chop flux"] = [beam_cross_flux, beam_cross_err,
                                      {"c": "k", "ls": "--", "twin_axes": True}]
        plt.close(plot_beam_ts(
                plot_dict, title="%s time series" % write_header.split("/")[-1],
                pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                plot_show=plot_show, plot_save=plot_save,
                write_header="%s_ts" % write_header, orientation=ORIENTATION))

    if cross:
        result = (beam_cross_flux, beam_cross_err, beam_cross_wt)
    else:
        result = (beam_flux, beam_err, beam_wt)
    if return_ts:
        result += (beam_use,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def make_raster(beams_flux, beams_err=None, write_header=None, pix_flag_list=[],
                raster_shape=ZPOLDBIG_SHAPE, return_pix_flag_list=False,
                plot=False, reg_interest=None, plot_show=False, plot_save=False,
                raster_thre=RASTER_THRE):
    """
    format the last dimension of the input object recording beams flux into 2-d
    raster, then plot the raster; the raster always starts from the lower left,
    then swipe to the right, move upwards by one, swiping to the left and so on,
    zigzaging to the top row.

    :param beams_flux: Obs or ObsArray, with flux of all the beams in the last
        dimension
    :type beams_flux: Union[Obs, ObsArray]
    :param beams_err: Obs or ObsArray, with error of all the beams in the last
        dimension, used for flagging pixels
    :type beams_err: Union[Obs, ObsArray]
    :param str write_header: str, full path to the title to save figures,
        if left None, will write to current folder with {obs_id} as the title
    :param list pix_flag_list: list, a list including pixels to be flagged, will
        be combined with auto flagged pixels in making figures and in the returned
        pix_flag_list, additional pixel flagging will be performed based on S/N
    :param list raster_shape: list, (az_len, alt_len), dimensional size in the
        azimuthal (horizontal) and elevation (vertical) directions of raster
    :param bool return_pix_flag_list: bool, bool, flag whether to return
        pix_flag_list
    :param bool plot: bool, flag whether to make figure
    :param dict reg_interest: dict, region of interest of array passed to
        ArrayMap.take_where() for plotting
    :param bool plot_show: bool, flag whether to call plt.show() and show figure
    :param bool plot_save: bool, flag whether to save the figure
    :param raster_thre: int or float, threshold of SNR of the pixel for it not
        to be flagged, by default use the value of RASTER_THRE
    :type raster_thre: Union[int, float]
    :return: tuple (raster_flux, [pix_flag_list]), are (ObsArray recording the
        raster flux reshaped in the last two dimensions. [optional, list of auto
        flagged pixels])
    :rtype: tuple
    """

    beams_flux_obs_array = ObsArray(beams_flux.copy())
    array_map = beams_flux_obs_array.array_map_
    raster_len = np.prod(raster_shape, dtype=int)
    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), beams_flux_obs_array.obs_id_)

    if beams_err is not None:  # auto flag pixels without anything at SNR > 3
        beams_err_obs_array = ObsArray(beams_err)
        beam_flux_med = weighted_proc_along_axis(
                beams_flux_obs_array, method="nanmedian",
                weight=1 / beams_err_obs_array ** 2, axis=-1)[0]
        pix_flag = np.all(
                ~((abs(beams_flux_obs_array - beam_flux_med).data_ > raster_thre *
                   beams_err_obs_array.data_) &
                  (abs(beams_flux_obs_array - beam_flux_med).data_ > raster_thre *
                   beams_err_obs_array.proc_along_time("nanmedian").data_)),
                axis=-1)
        pix_flag_list = pix_flag_list.copy() + array_map.take_by_flag(
                pix_flag).array_idxs_.tolist()
        pix_flag_list = np.unique(pix_flag_list, axis=0).tolist()

    if beams_flux_obs_array.len_ < raster_len:  # add placeholder beam
        num_fill = raster_len - beams_flux_obs_array.len_
        beams_flux_obs_array.append(beams_flux_obs_array.replace(
                arr_in=np.full(beams_flux_obs_array.shape_[:-1] + (num_fill,),
                               fill_value=np.nan),
                ts=(None if beams_flux_obs_array.ts_.empty_flag_ else
                    np.arange(1, num_fill + 1) * beams_flux_obs_array.ts_.interv_),
                chop=(None if beams_flux_obs_array.chop_.empty_flag_ else np.tile(
                        beams_flux_obs_array.chop_.data_[-1:], num_fill)),
                obs_id_arr=np.char.add(
                        "empty_beam_", np.arange(num_fill).astype(str))))
    elif beams_flux_obs_array.len_ > raster_len:
        warnings.warn("beam number grater than raster size.", UserWarning)
        beams_flux_obs_array = \
            beams_flux_obs_array.take_by_idx_along_time(range(raster_len))

    raster_flux = beams_flux_obs_array.replace(arr_in=np.where(  # rearrange shpe
            np.arange(raster_shape[1])[:, None] % 2,
            beams_flux_obs_array.data_.reshape(
                    *beams_flux_obs_array.shape_[:-1],
                    *raster_shape[::-1])[..., ::-1],
            beams_flux_obs_array.data_.reshape(
                    *beams_flux_obs_array.shape_[:-1], *raster_shape[::-1])),
            chop=None, ts=None)

    if plot:
        fig = FigArray.init_by_array_map(
                array_map if reg_interest is None else array_map.take_where(
                        **reg_interest), orientation=ORIENTATION,
                x_size=0.5, y_size=.5, axs_fontsize=2)
        fig.imshow(raster_flux, origin="lower")
        fig.imshow_flag(pix_flag_list=pix_flag_list)
        fig.set_xlabel("azimuth")
        fig.set_ylabel("altitude")
        fig.set_labels(beams_flux, orientation=ORIENTATION)
        fig.set_title(title="%s raster" % write_header.split("/")[-1])
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig("%s_raster.png" % write_header)
        plt.close(fig)

    result = (raster_flux,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def stack_raster(raster, raster_wt=None, write_header=None, pix_flag_list=[],
                 plot=False, plot_show=False, plot_save=False):
    """
    Stack the raster along spatial dimension to get a high SNR raster, used for
    analyzing data taken with zpold or zpoldbig

    :param ObsArray raster: ObsArray, object containing the raster in the last
        two dimensions
    :param raster_wt: weight of the raster, raster * raster_wt will be stacked
    :type raster_wt: Union[int, float, numpy.ndarray, Obs, ObsArray]
    :param str write_header: str, full path to the title to save files/figures,
        if left None, will write to current folder with {obs_id}_raster_stack
        as file header
    :param list pix_flag_list: list, a list including pixels to be flagged, these
        pixels will not be used in stacking
    :param bool plot: bool, flag whether to make the figure of the stacked raster
    :param bool plot_show: bool, flag whether to show the figure
    :param bool plot_save: bool, flag whether to save the figure
    :return: ObsArray object containing the stacked raster
    :rtype: ObsArray
    """

    raster_norm = ObsArray(raster * raster_wt)
    array_map = raster_norm.array_map_
    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), raster_norm.obs_id_)

    stacked_raster = ObsArray()  # stack raster
    for spat in range(array_map.array_spat_llim_,
                      array_map.array_spat_ulim_ + 1):
        spat_raster = raster_norm.exclude_where(spat_spec_list=pix_flag_list). \
            take_where(spat=spat)
        if spat_raster.len_ > 0:
            stacked_pix = spat_raster.proc_along_axis(
                    method="nanmean", axis=0, array_map=[[spat, 0, spat, 0]])
            stacked_raster.expand(stacked_pix)

    if plot:
        fig = FigArray.init_by_array_map(stacked_raster, orientation=ORIENTATION,
                                         x_size=0.5, y_size=.5, axs_fontsize=2)
        fig.imshow(stacked_raster, origin="lower")
        fig.set_xlabel("azimuth")
        fig.set_ylabel("altitude")
        fig.set_labels(raster, orientation=ORIENTATION)
        fig.set_title(title="%s stacked raster" % write_header.split("/")[-1])
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig("%s_raster_stack.png" % write_header)
        plt.close(fig)

    return stacked_raster


def read_beam(file_header, array_map=None, obs_log=None, flag_ts=True,
              is_flat=False):
    """
    function to read MCE time series data, convert to array map layout, add
    auxiliary information from obs log, and flag outliers if opted

    :param str file_header: str, full path to the data file
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    :param ObsLog obs_log: ObsLog, optional, if not None, will try to find the
        entry in the provided obs_log and add to the obs_info of the output obj
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :return: Obs or ObsArray object containing the data
    :rtype: Union[Obs, ObsArray]
    """

    try:
        beam = Obs.read_header(filename=file_header)  # read in data
    except Exception:
        warnings.warn("fail to read in %s." % file_header, UserWarning)
        beam = Obs(obs_id=file_header.split("/")[-1])
    if array_map is not None:  # transform into ObsArray
        if beam.empty_flag_:
            beam = ObsArray(
                    np.empty((array_map.len_, 0), dtype=ObsArray.dtype_),
                    array_map=array_map)
            beam.empty_flag_ = True
        else:
            beam = beam.to_obs_array(array_map=array_map)
    if flag_ts and (not beam.empty_flag_):
        beam = auto_flag_ts(beam, is_flat=is_flat)
    if (obs_log is not None) and (len(obs_log) > 0) and (not beam.empty_flag_):
        with warnings.catch_warnings():
            if is_flat:
                warnings.filterwarnings("ignore", message=
                "No entry is found in obs log.")
            beam.match_obs_log(obs_log)  # find entry in obs_log

    return beam


def read_beam_pair(file_header1, file_header2, array_map=None, obs_log=None,
                   flag_ts=True, is_flat=False, match_same_phase=MATCH_SAME_PHASE,
                   stack_factor=STACK_FACTOR):
    """
    function to read in data of a beam pair with read_beam(), then stack the
    time series of beam pair using get_match_phase_obs() with given
    match_same_phase and stack_method. The returned stacked beam pair will
    be (matched_beam1 + stack_factor * matched_beam2)/2

    :param str file_header1: str, full path to the data file of the first beam
    :param str file_header2: str, full path to the data file of the second beam
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process, passed to read_beam()
    :param ObsLog obs_log: ObsLog, optional, if not None, will try to find the
        entry in the provided obs_log and add to the obs_info of the output obj,
        passed to read_beam()
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :param bool match_same_phase: bool, flag whether to match chop chunks of the
        same chop phase, if False will match chunks of the opposite phase,
        default use MATCH_SAME_PHASE, passed to get_match_phase_obs()
    :param stack_factor: int or float, the factor to applied to the second beam
        when stack the beams; 1 means the two beams will be added together, -1
        means the second beam will be subtracted from the first; use STACK_FACTOR
        by default
    :type stack_factor: Union[int, float]
    :return: Obs or ObsArray object containing the stacked beam pair
    :rtype: Union[Obs, ObsArray]
    """

    beam1, beam2 = [read_beam(file_header, array_map=array_map, obs_log=obs_log,
                              flag_ts=flag_ts, is_flat=is_flat)
                    for file_header in (file_header1, file_header2)]
    matched_beam1, matched_beam2 = get_match_phase_obs(
            beam1, beam2, match_same_phase=match_same_phase)
    stacked_beam_pair = (matched_beam1 + stack_factor * matched_beam2) / 2
    if flag_ts:
        stacked_beam_pair = auto_flag_ts(stacked_beam_pair, is_flat=is_flat)

    return stacked_beam_pair


def reduce_beam(file_header, write_dir=None, write_suffix="", array_map=None,
                obs_log=None, is_flat=False, pix_flag_list=[], flat_flux=1,
                flat_err=0, cross=False, do_desnake=False, ref_pix=None,
                do_smooth=False, do_ica=False, spat_excl=None, return_ts=False,
                return_pix_flag_list=False, plot=False, plot_ts=False,
                reg_interest=None, plot_flux=False, plot_show=False,
                plot_save=False):
    """
    a wrapper function to read data and reduce beam in the standard way
    """

    if write_dir is None:
        write_dir = os.getcwd()
    write_suffix = str(write_suffix)
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_ica),
                            ("desnake", "smooth", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method

    print("Processing beam %s." % file_header)
    beam = read_beam(file_header=file_header, array_map=array_map,
                     obs_log=obs_log, flag_ts=True, is_flat=is_flat)  # read data
    if (not beam.empty_flag_) and beam.len_ > 0:
        write_header = os.path.join(write_dir, beam.obs_id_ + write_suffix)
        result = proc_beam(
                beam, write_header=write_header, is_flat=is_flat,
                pix_flag_list=pix_flag_list, flat_flux=flat_flux, flat_err=flat_err,
                cross=cross, do_desnake=do_desnake, ref_pix=ref_pix,
                do_smooth=do_smooth, do_ica=do_ica, spat_excl=spat_excl,
                return_ts=return_ts, return_pix_flag_list=return_pix_flag_list,
                plot=plot, plot_ts=plot_ts, reg_interest=reg_interest,
                plot_flux=plot_flux, plot_show=plot_show, plot_save=plot_save)
    else:
        result = (beam.copy(), beam.copy(), beam.copy())
        if return_ts:
            result += (beam,)
        if return_pix_flag_list:
            result += (pix_flag_list,)

    return result


def reduce_beam_pair(file_header1, file_header2, write_dir=None, write_suffix="",
                     array_map=None, obs_log=None, is_flat=False, pix_flag_list=[],
                     flat_flux=1, flat_err=0, do_desnake=False, ref_pix=None,
                     do_smooth=False, do_ica=False, spat_excl=None,
                     return_ts=False, return_pix_flag_list=False, plot=False,
                     plot_ts=False, reg_interest=None, plot_flux=False,
                     plot_show=False, plot_save=False):
    """
    a wrapper function to read data and reduce beam pair in the standard way
    """

    if write_dir is None:
        write_dir = os.getcwd()
    write_suffix = str(write_suffix)
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_ica),
                            ("desnake", "smooth", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method

    print("Processing beam pair %s and %s." % (file_header1, file_header2))
    beam_pair = read_beam_pair(file_header1=file_header1, file_header2=file_header2,
                               array_map=array_map, obs_log=obs_log,
                               flag_ts=True, is_flat=is_flat)
    if len(beam_pair.obs_id_list_) == 1:
        header = beam_pair.obs_id_
    else:
        header = beam_pair.obs_id_list_[0]
        split_header1, split_header2 = beam_pair.obs_id_list_[0].split("_"), \
                                       beam_pair.obs_id_list_[1].split("_")
        idx = 0
        while (idx < min(len(split_header1), len(split_header2))) and \
                (split_header1[idx] == split_header2[idx]):
            idx += 1
        if idx < len(split_header2):
            header += "-" + "_".join(split_header2[idx:])

    if (not beam_pair.empty_flag_) and (beam_pair.len_ > 0):
        write_header = os.path.join(write_dir, header + write_suffix)
        result = proc_beam(
                beam_pair, write_header=write_header, is_flat=is_flat,
                pix_flag_list=pix_flag_list, flat_flux=flat_flux, flat_err=flat_err,
                do_desnake=do_desnake, ref_pix=ref_pix, do_smooth=False,
                do_ica=do_ica, spat_excl=spat_excl, return_ts=return_ts,
                return_pix_flag_list=return_pix_flag_list, plot=plot,
                plot_ts=plot_ts, reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save)
    else:
        result = (beam_pair.copy(), beam_pair.copy(), beam_pair.copy())
        if return_ts:
            result += (beam_pair,)
        if return_pix_flag_list:
            result += (pix_flag_list,)

    return result


def read_beams(file_header_list, array_map=None, obs_log=None, flag_ts=True,
               is_flat=False, parallel=False):
    """
    Function to read multiple MCE data files, optionally in a parallelized mode.
    It takes ~ 5 min on spectrosaurus to read 700 beams in parallel. It is
    recommended to clean the memory or start a new thread before running it in
    parallel.

    :param dict file_header_list: list, of the paths to the headers of the data
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    :param ObsLog obs_log: ObsLog, optional, if not None, will try to find the
        entry in the provided obs_log and add to the obs_info of the output obj
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :param bool parallel: bool, flag whether to run it in parallelized mode,
        would accelerate the process by many factors on a multi-core machine
    :return: Obs or ObsArray object containing all the data concatenated
    :rtype: Union[Obs, ObsArray]
    """

    args_list = []  # build variable list for read_beam
    for file_header in file_header_list:
        args = ()
        for var_name in inspect.getfullargspec(read_beam)[0]:
            args += (locals()[var_name],)
        args_list.append(args)

    if parallel and check_parallel():
        gc.collect()
        with multiprocessing.get_context("fork").Pool(
                min(MAX_THREAD_NUM, len(args_list))) as pool:
            results = pool.starmap(read_beam, args_list)
    else:
        results = []
        for args in args_list:
            results += [read_beam(*args)]

    kwargs = {}
    if array_map is not None:  # combine all beams
        type_result = ObsArray
        kwargs["array_map"] = array_map
    else:
        type_result = Obs
    data_list, ts_list, chop_list, obs_id_list, obs_id_arr_list, \
    obs_info_list = [], [], [], [], [], []
    for beam in results:
        if not beam.empty_flag_:
            data_list.append(beam.data_)
            ts_list.append(beam.ts_.data_)
            chop_list.append(beam.chop_.data_)
            obs_id_list += beam.obs_id_list_
            obs_id_arr_list.append(beam.obs_id_arr_.data_)
            obs_info_list.append(beam.obs_info_.table_)
    kwargs["arr_in"] = np.concatenate(data_list, axis=-1)
    kwargs["ts"] = np.concatenate(ts_list)
    kwargs["chop"] = np.concatenate(chop_list)
    kwargs["obs_id_list"] = obs_id_list
    kwargs["obs_id_arr"] = np.concatenate(obs_id_arr_list)
    kwargs["obs_info"] = vstack(obs_info_list, join_type="outer")
    kwargs["obs_id"] = obs_id_list[0]
    all_beams = type_result(**kwargs)

    return all_beams


# ====================== high level reduction functions ========================


def reduce_beams(data_header, data_dir=None, write_dir=None, write_suffix="",
                 array_map=None, obs_log=None, is_flat=False, pix_flag_list=[],
                 flat_flux=1, flat_err=0, cross=False, parallel=False,
                 do_desnake=False, ref_pix=None, do_smooth=False, do_ica=False,
                 spat_excl=None, return_ts=False, return_pix_flag_list=False,
                 plot=False, plot_ts=False, reg_interest=None, plot_flux=False,
                 plot_show=False, plot_save=False):
    """
    reduce the data of beam in data_header, and return the flux of beams
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    args_list = []  # build variable list for reduce_beam()
    flat_flux_group_flag, flat_err_group_flag = False, False
    if isinstance(flat_flux, (Obs, ObsArray)) and flat_flux.len_ > 1:
        flat_flux_group_flag, flat_flux_group = True, flat_flux
    if isinstance(flat_err, (Obs, ObsArray)) and flat_err.len_ > 1:
        flat_err_group_flag, flat_err_group = True, flat_err
    i = 0
    for header in data_header:
        for idxs in data_header[header]:
            if flat_flux_group_flag:
                flat_flux = flat_flux_group.take_by_idx_along_time(i)
            if flat_err_group_flag:
                flat_err = flat_err_group.take_by_idx_along_time(i)
            i += 1
            for beam_idx in range(idxs[0], idxs[1] + 1):
                file_header = os.path.join(
                        data_dir, "%s_%04d" % (header, beam_idx))
                args = ()
                for var_name in inspect.getfullargspec(reduce_beam)[0]:
                    args += (locals()[var_name],)
                args_list.append(args)

    if parallel and check_parallel():
        gc.collect()
        with multiprocessing.get_context("fork").Pool(
                min(MAX_THREAD_NUM, len(args_list))) as pool:
            results = pool.starmap(reduce_beam, args_list)
    else:
        results = []
        for args in args_list:
            results += [reduce_beam(*args)]

    type_result = Obs if array_map is None else ObsArray
    beams_flux, beams_err, beams_wt = type_result(), type_result(), type_result()
    if return_ts:
        beams_ts = type_result()
    for result in results:
        if result[0].empty_flag_ and result[1].empty_flag_:
            nan_beam_obs_id = result[0].obs_id_ if result[0].obs_id_ != "0" \
                else "failed to read"
            nan_beam = beams_flux.replace(
                    arr_in=np.full(beams_flux.shape_[:-1] + (1,),
                                   fill_value=np.nan), ts=[result[0].ts_.t_mid_],
                    chop=[False], obs_id=nan_beam_obs_id,
                    obs_id_list=[nan_beam_obs_id],
                    obs_id_arr=np.array([nan_beam_obs_id]),
                    obs_info=result[0].obs_info_)
            beams_flux.append(nan_beam)
            beams_err.append(nan_beam)
            beams_wt.append(nan_beam)
        else:
            beams_flux.append(result[0])
            beams_err.append(result[1])
            beams_wt.append(result[2])
        if return_ts:
            beams_ts.append(result[3])
        if return_pix_flag_list:
            pix_flag_list = pix_flag_list.copy() + result[-1]

    result = (beams_flux, beams_err, beams_wt)
    if return_ts:
        result += (beams_ts,)
    if return_pix_flag_list:
        result += (np.unique(pix_flag_list, axis=0).tolist(),)

    return result


def reduce_beam_pairs(data_header, data_dir=None, write_dir=None,
                      write_suffix="", array_map=None, obs_log=None,
                      is_flat=False, pix_flag_list=[], flat_flux=1, flat_err=0,
                      parallel=False, do_desnake=False, ref_pix=None,
                      do_smooth=False, do_ica=False, spat_excl=None,
                      return_ts=False, return_pix_flag_list=False, plot=False,
                      plot_ts=False, reg_interest=None, plot_flux=False,
                      plot_show=False, plot_save=False, use_hk=True):
    """
    reduce the data files in data_header by callling reduce_beam_pair() which
    stack each beam pair, and return the flux, error and weight of the beam pairs

    :raises RunTimeError: no beam pair is matched
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    args_list = []  # build variable list for reduce_beam_pair
    flat_flux_group_flag, flat_err_group_flag = False, False
    if isinstance(flat_flux, (Obs, ObsArray)) and flat_flux.len_ > 1:
        flat_flux_group_flag, flat_flux_group = True, flat_flux
    if isinstance(flat_err, (Obs, ObsArray)) and flat_err.len_ > 1:
        flat_err_group_flag, flat_err_group = True, flat_err
    i = 0
    for header in data_header:
        for idxs in data_header[header]:
            if flat_flux_group_flag:
                flat_flux = flat_flux_group.take_by_idx_along_time(i)
            if flat_err_group_flag:
                flat_err = flat_err_group.take_by_idx_along_time(i)
            i += 1
            beams_info = Obs()
            for beam_idx in range(idxs[0], idxs[1] + 1):
                file_header = os.path.join(
                        data_dir, "%s_%04d" % (header, beam_idx))
                beam_info = Obs.read_header(
                        filename=file_header, try_data=False, try_ts=False,
                        try_chop=False).replace(arr_in=[0])
                beams_info.append(beam_info)
            nod = (np.arange(idxs[1] - idxs[0] + 1) % 2).astype(Chop.dtype_)
            if (NOD_COLNAME in beams_info.obs_info_.table_.colnames) and \
                    ("obs_id" in beams_info.obs_info_.table_.colnames) and \
                    use_hk:
                for idx, obs_id in enumerate(beams_info.obs_id_arr_.data_):
                    if obs_id in beams_info.obs_info_.table_["obs_id"]:
                        tb_idx = np.nonzero(
                                obs_id == beams_info.obs_info_.table_["obs_id"]
                        )[0][0]
                        if not beams_info.obs_info_.table_[NOD_COLNAME]. \
                                mask[tb_idx]:
                            nod[idx] = \
                                beams_info.obs_info_.table_[NOD_COLNAME][tb_idx]
            beams_info.update_chop(nod)
            beams_right, beams_left = get_match_phase_pair(beams_info)

            for header1, header2 in zip(beams_left.obs_id_arr_.data_,
                                        beams_right.obs_id_arr_.data_):
                if NOD_PHASE == 1:
                    file_header1, file_header2 = \
                        os.path.join(data_dir, header1), \
                        os.path.join(data_dir, header2)
                elif NOD_PHASE == -1:
                    file_header1, file_header2 = \
                        os.path.join(data_dir, header2), \
                        os.path.join(data_dir, header1)
                else:
                    raise ValueError("NOD_PHASE is unknown")
                args = ()
                for var_name in inspect.getfullargspec(reduce_beam_pair)[0]:
                    args += (locals()[var_name],)
                args_list.append(args)
        if len(args_list) == 0:
            raise RuntimeError("No beam pair is matched, may not be nodding.")

    if parallel and check_parallel():
        gc.collect()
        with multiprocessing.get_context("fork").Pool(
                min(MAX_THREAD_NUM, len(args_list))) as pool:
            results = pool.starmap(reduce_beam_pair, args_list)
    else:
        results = []
        for args in args_list:
            results += [reduce_beam_pair(*args)]

    type_result = Obs if array_map is None else ObsArray
    beam_pairs_flux, beam_pairs_err, beam_pairs_wt = \
        type_result(), type_result(), type_result()
    if return_ts:
        beam_pairs_ts = type_result()
    for result in results:
        if result[0].empty_flag_ and result[1].empty_flag_:
            nan_beam_pair_obs_id = result[0].obs_id_ if \
                result[0].obs_id_ != "0" else "failed to read"
            nan_beam_pair = beam_pairs_flux.replace(
                    arr_in=np.full(beam_pairs_flux.shape_[:-1] + (1,),
                                   fill_value=np.nan), ts=result[0].ts_.t_mid_,
                    chop=[False], obs_id=nan_beam_pair_obs_id,
                    obs_id_list=[nan_beam_pair_obs_id],
                    obs_id_arr=np.array([nan_beam_pair_obs_id]),
                    obs_info=result[0].obs_info_)
            beam_pairs_flux.append(nan_beam_pair)
            beam_pairs_err.append(nan_beam_pair)
            beam_pairs_wt.append(nan_beam_pair)
        else:
            beam_pairs_flux.append(result[0])
            beam_pairs_err.append(result[1])
            beam_pairs_wt.append(result[2])
        if return_ts:
            beam_pairs_ts.append(result[3])
        if return_pix_flag_list:
            pix_flag_list = pix_flag_list.copy() + result[-1]

    result = (beam_pairs_flux, beam_pairs_err, beam_pairs_wt)
    if return_ts:
        result += (beam_pairs_ts,)
    if return_pix_flag_list:
        result += (np.unique(pix_flag_list, axis=0).tolist(),)

    return result


def reduce_skychop(flat_header, data_dir=None, write_dir=None, write_suffix="",
                   array_map=None, obs_log=None, pix_flag_list=[], parallel=False,
                   return_ts=False, return_pix_flag_list=True, table_save=True,
                   plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                   plot_show=False, plot_save=True, analyze=False):
    """
    process data taken as skychop
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    result = reduce_beams(
            flat_header, data_dir=data_dir, write_dir=write_dir,
            write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
            is_flat=True, pix_flag_list=pix_flag_list, parallel=parallel,
            return_ts=(return_ts or (plot and plot_ts) or analyze),
            return_pix_flag_list=True, plot=plot, plot_ts=False,
            reg_interest=reg_interest, plot_flux=plot_flux,
            plot_show=plot_show, plot_save=plot_save)
    flat_beams_flux, flat_beams_err, flat_beams_wt = result[:3]
    if return_ts or (plot and plot_ts) or analyze:
        flat_beams_ts = result[3]
    pix_flag_list = result[-1]

    beam_num_list = [idxs[1] - idxs[0] + 1 for header in flat_header
                     for idxs in flat_header[header]]

    type_result = Obs if array_map is None else ObsArray
    flat_flux, flat_err, flat_wt = type_result(), type_result(), type_result()
    for idx_i, idx_e in zip(np.cumsum([0] + beam_num_list[:-1]),
                            np.cumsum(beam_num_list)):
        group_flux, group_err_ex, group_wt = weighted_proc_along_axis(
                flat_beams_flux.take_by_idx_along_time(range(idx_i, idx_e)),
                weight=None)
        group_err_in = ((flat_beams_err ** 2).proc_along_time(
                method="nanmean")).sqrt()
        group_err = group_err_ex.replace(
                arr_in=np.choose(group_err_ex.data_ < group_err_in.data_,
                                 [group_err_ex.data_, group_err_in.data_]))
        flat_flux.append(group_flux)
        flat_err.append(group_err)
        flat_wt.append(group_wt)

    flat_file_header = build_header(flat_header)
    if table_save:  # save to csv
        for obs, name in zip((flat_beams_flux, flat_beams_err, flat_flux, flat_err),
                             ("beams_flux", "beams_err", "flux", "err")):
            obs.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_%s.csv" % (flat_file_header, name)),
                    overwrite=True)
        flat_beams_flux.obs_info_.table_.write(os.path.join(
                write_dir, "%s_beams_info.csv" % flat_file_header),
                overwrite=True)
    if plot:  # plot time series and flux
        plot_dict = {"flux": (flat_flux, flat_err, {"c": "k"}),
                     "beam flux": (flat_beams_flux, flat_beams_err, {"c": "y"})}
        if plot_ts:
            plot_dict["raw data"] = (flat_beams_ts, {"twin_axes": True})
        plt.close(plot_beam_ts(
                plot_dict, title=(flat_file_header + " beam flux" +
                                  " and time series" * plot_ts),
                pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                plot_show=plot_show, plot_save=plot_save,
                write_header=os.path.join(
                        write_dir, "%s_beams_flux" % flat_file_header),
                orientation=ORIENTATION))

    if analyze:
        beams_rms = analyze_performance(
                flat_beams_ts, write_header=
                os.path.join(write_dir, flat_file_header),
                pix_flag_list=pix_flag_list, plot=plot, plot_rms=plot_flux,
                plot_ts=False, reg_interest=reg_interest, plot_psd=plot_ts,
                plot_specgram=False, plot_show=plot_show, plot_save=plot_save)
        if table_save:
            beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_rms.csv" % flat_file_header), overwrite=True)

    result = (flat_flux, flat_err, flat_wt)
    if return_ts:
        result += (flat_beams_ts,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def reduce_zobs(data_header, data_dir=None, write_dir=None, write_suffix="",
                array_map=None, obs_log=None, pix_flag_list=[], flat_flux=1,
                flat_err=0, parallel=False, stack=False, do_desnake=False,
                ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                return_ts=False, return_pix_flag_list=True, table_save=True,
                plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                plot_show=False, plot_save=True, analyze=False, use_hk=True):
    """
    reduce the data from zobs command

    :param bool use_hk: bool, flag whether to use hk file as nodding phase
    :raises RunTimeError: not nodding
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_ica),
                            ("desnake", "smooth", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method
    plot_dict = {}
    if not stack:
        if do_ica:
            warnings.warn(
                    "will perform ICA decomposition w/o stacking beam pairs!")
        result = reduce_beams(
                data_header=data_header, data_dir=data_dir, write_dir=write_dir,
                write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
                is_flat=False, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
                flat_err=flat_err, parallel=parallel, do_desnake=do_desnake,
                ref_pix=ref_pix, do_smooth=do_smooth, do_ica=do_ica,
                spat_excl=spat_excl, return_ts=return_ts | analyze,
                return_pix_flag_list=True, plot=plot,
                plot_ts=plot_ts, reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save)
        beams_flux, beams_err, beams_wt = result[:3]
        plot_dict["beam flux"] = [beams_flux, beams_err,
                                  {"c": "y", "ls": ":", "lw": 0.5}]
        beam_num_list = [idxs[1] - idxs[0] + 1 for header in data_header
                         for idxs in data_header[header]]
        nod = np.empty(0, dtype=Chop.dtype_)
        for beam_num in beam_num_list:  # default nodding phase
            nod = np.concatenate((nod, (
                    np.arange(beam_num) % 2).astype(Chop.dtype_)))
        if (NOD_COLNAME in beams_flux.obs_info_.table_.colnames) and \
                ("obs_id" in beams_flux.obs_info_.table_.colnames) and use_hk:
            for idx, obs_id in enumerate(beams_flux.obs_id_arr_.data_):
                if obs_id in beams_flux.obs_info_.table_["obs_id"]:
                    tb_idx = np.nonzero(
                            obs_id == beams_flux.obs_info_.table_["obs_id"])[0][0]
                    if not beams_flux.obs_info_.table_[NOD_COLNAME].mask[tb_idx]:
                        nod[idx] = \
                            beams_flux.obs_info_.table_[NOD_COLNAME][tb_idx]
        if (nod.sum() == 0) or (~nod.sum() == 0):
            raise RuntimeError("The data is not nodding.")
        beams_flux.update_chop(nod)
        beams_err.update_chop(nod)
        beams_wt.update_chop(nod)

        type_result = type(beams_flux)
        beam_pairs_flux, beam_pairs_err, beam_pairs_wt = \
            type_result(), type_result(), type_result()
        for idx_i, idx_e in zip(np.cumsum([0] + beam_num_list[:-1]),
                                np.cumsum(beam_num_list)):
            beam_flux_r, beam_flux_l = get_match_phase_pair(
                    beams_flux.take_by_idx_along_time(range(idx_i, idx_e)))
            beam_err_r, beam_err_l = get_match_phase_pair(
                    beams_err.take_by_idx_along_time(range(idx_i, idx_e)))
            beam_wt_r, beam_wt_l = get_match_phase_pair(
                    beams_wt.take_by_idx_along_time(range(idx_i, idx_e)))
            group_flux = (beam_flux_l - beam_flux_r) / 2 * NOD_PHASE
            group_err = (beam_err_l ** 2 + beam_err_r ** 2).sqrt() / 2
            group_wt = beam_wt_l + beam_wt_r
            group_time = (beam_flux_l.ts_ + beam_flux_r.ts_) / 2
            for group in (group_flux, group_err, group_wt):
                group.update_ts(group_time)
            for beam_pair, group in zip(
                    (beam_pairs_flux, beam_pairs_err, beam_pairs_wt),
                    (group_flux, group_err, group_wt)):
                beam_pair.append(group)
    else:
        result = reduce_beam_pairs(
                data_header=data_header, data_dir=data_dir, write_dir=write_dir,
                write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
                is_flat=False, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
                flat_err=flat_err, parallel=parallel, do_desnake=do_desnake,
                ref_pix=ref_pix, do_smooth=do_smooth, do_ica=do_ica,
                spat_excl=spat_excl, return_ts=return_ts | analyze,
                return_pix_flag_list=True, plot=plot, plot_ts=plot_ts,
                reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save, use_hk=use_hk)
        beam_pairs_flux, beam_pairs_err, beam_pairs_wt = result[:3]
    plot_dict["beam pair flux"] = (beam_pairs_flux, beam_pairs_err, {"c": "k"})
    if return_ts or analyze:
        zobs_ts = result[3]
    pix_flag_list = result[-1]

    zobs_flux, zobs_err_ex, zobs_wt = weighted_proc_along_axis(  # flux of zobs
            beam_pairs_flux, weight=1 / beam_pairs_err ** 2)
    zobs_err_in = (beam_pairs_err ** 2).proc_along_time("nanmean").sqrt() / \
                  beam_pairs_flux.proc_along_time("num_is_finite").sqrt()
    zobs_err = zobs_err_ex.replace(
            arr_in=np.choose(zobs_err_ex.data_ < zobs_err_in.data_,
                             [zobs_err_ex.data_, zobs_err_in.data_]))
    pix_flag_list = auto_flag_pix_by_flux(zobs_flux, zobs_err,
                                          pix_flag_list=pix_flag_list)

    data_file_header = build_header(data_header) + write_suffix
    if table_save:
        tb_list = [beam_pairs_flux, beam_pairs_err, zobs_flux, zobs_err]
        tb_names = ["beam_pairs_flux", "beam_pairs_err", "flux", "err"]
        if not stack:
            tb_list += [beams_flux, beams_err]
            tb_names += ["beams_flux", "beams_err"]
            beams_flux.obs_info_.table_.write(os.path.join(
                    write_dir, "%s_beams_info.csv" % data_file_header),
                    overwrite=True)
        for obs, name in zip(tb_list, tb_names):
            obs.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_%s.csv" % (data_file_header, name)),
                    overwrite=True)
        beam_pairs_flux.obs_info_.table_.write(os.path.join(
                write_dir, "%s_beam_pairs_info.csv" % data_file_header),
                overwrite=True)
    if plot:
        plt.close(plot_beam_flux(
                zobs_flux, title=data_file_header + " zobs flux",
                pix_flag_list=pix_flag_list, plot_show=plot_show,
                plot_save=plot_save, write_header=os.path.join(
                        write_dir, "%s_flux" % data_file_header)))
        plt.close(plot_beam_flux(
                zobs_err, title=data_file_header + " zobs flux",
                pix_flag_list=pix_flag_list, plot_show=plot_show,
                plot_save=plot_save, write_header=os.path.join(
                        write_dir, "%s_err" % data_file_header)))

        zobs_flux_array = ObsArray(zobs_flux)  # plot spectrum
        array_map = zobs_flux_array.array_map_
        fig = FigSpec.plot_spec(zobs_flux, yerr=zobs_err,
                                pix_flag_list=pix_flag_list, color="k")
        fig.imshow_flag(pix_flag_list=pix_flag_list)
        fig.plot_all_spat(
                [array_map.array_spec_llim_, array_map.array_spec_ulim_],
                [0, 0], "k:")
        fig.set_title("%s spectrum" % data_file_header)
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(os.path.join(write_dir, "%s_spec.png" % data_file_header))
        plt.close(fig)

        # get PWV
        if ("UTC" in zobs_flux.obs_info_.table_.colnames) and \
                ("mm PWV" in zobs_flux.obs_info_.table_.colnames):
            tb_use = zobs_flux.obs_info_.table_[
                ~zobs_flux.obs_info_.table_.mask["UTC"]]
            tb_use.sort("UTC")
            t_arr = Time.strptime(tb_use["UTC"],
                                  format_string="%Y-%m-%dU%H:%M:%S"). \
                to_value(format="unix")
            t_arr += tb_use["Scan duration"] / 2
            pwv_arr = tb_use["mm PWV"]
            obs_pwv = beam_pairs_flux.replace(
                    arr_in=np.tile(pwv_arr, beam_pairs_flux.shape_[:-1] + (1,)),
                    ts=t_arr, chop=None)
            plot_dict["PWV"] = [obs_pwv, {"ls": "--", "twin_axes": True,
                                          "c": "r", "marker": ".",
                                          "markersize": 3}]

        if not stack:
            plt.close(plot_beam_ts(
                    plot_dict, title=(data_file_header + " beam flux"),
                    pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                    plot_show=plot_show, plot_save=plot_save,
                    write_header=os.path.join(
                            write_dir, "%s_beams_flux" % data_file_header),
                    orientation=ORIENTATION))
            plot_dict.pop("beam flux")
        if beam_pairs_flux.len_ > 1:  # cumulative flux measurement
            type_result = type(beam_pairs_flux)
            cum_flux, cum_err, cum_wt = type_result(), type_result(), \
                                        type_result()
            for idx in range(beam_pairs_flux.len_):
                flux_use, err_use, wt_use = [
                    obs.take_by_idx_along_time(range(idx + 1)) for obs in
                    (beam_pairs_flux, beam_pairs_err, beam_pairs_wt)]
                flux, err_ex, wt = weighted_proc_along_axis(
                        flux_use, weight=1 / err_use ** 2)
                err_in = (err_use ** 2).proc_along_time("nanmean").sqrt() / \
                         flux_use.proc_along_time("num_is_finite").sqrt()
                err = err_ex.replace(
                        arr_in=np.choose(err_ex.data_ < err_in.data_,
                                         [err_ex.data_, err_in.data_]))
                flux.ts_ = beam_pairs_flux.take_by_idx_along_time(idx).ts_
                cum_flux.append(flux)
                cum_err.append(err)
                cum_wt.append(wt)
            cum_flux.ts_ += cum_flux.ts_.interv_ / 4
            plot_dict["cum flux"] = [cum_flux, cum_err,
                                     {"c": "c", "ls": ":", "lw": 1}]
        plt.close(plot_beam_ts(
                plot_dict, title=(data_file_header + " beam pair flux"),
                pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                plot_show=plot_show, plot_save=plot_save,
                write_header=os.path.join(
                        write_dir, "%s_beam_pairs_flux" % data_file_header),
                orientation=ORIENTATION))

    if analyze:
        beams_rms = analyze_performance(
                zobs_ts, write_header=os.path.join(
                        write_dir, data_file_header), pix_flag_list=pix_flag_list,
                plot=plot, plot_rms=plot_flux, plot_ts=False,
                reg_interest=reg_interest, plot_psd=plot_ts,
                plot_specgram=False, plot_show=plot_show, plot_save=plot_save)
        if table_save:
            beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_rms.csv" % data_file_header), overwrite=True)

    result = (zobs_flux, zobs_err, zobs_wt)
    if return_ts:
        result += (zobs_ts,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


# TODO: return intermediate result
# TODO: add write suffix automatically

def reduce_calibration(data_header, data_dir=None, write_dir=None,
                       write_suffix="", array_map=None, obs_log=None,
                       is_flat=False, pix_flag_list=[], flat_flux=1, flat_err=0,
                       cross=False, parallel=False, do_desnake=False,
                       ref_pix=None, do_smooth=False, do_ica=False,
                       spat_excl=None, return_ts=False,
                       return_pix_flag_list=True, table_save=True, plot=True,
                       plot_ts=True, reg_interest=None, plot_flux=True,
                       plot_show=False, plot_save=True, analyze=False):
    """
    reduce data for general calibration that does not involve nodding or raster,
    but just continuous chop observations like pointing or focus
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_ica),
                            ("desnake", "smooth", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method
    result = reduce_beams(
            data_header=data_header, data_dir=data_dir, write_dir=write_dir,
            write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
            is_flat=is_flat, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
            flat_err=flat_err, cross=cross, parallel=parallel,
            do_desnake=do_desnake, ref_pix=ref_pix, do_smooth=do_smooth,
            do_ica=do_ica, spat_excl=spat_excl, return_ts=return_ts | analyze,
            return_pix_flag_list=True, plot=plot, plot_ts=plot_ts,
            reg_interest=reg_interest, plot_flux=plot_flux, plot_show=plot_show,
            plot_save=plot_save)
    beams_flux, beams_err, beams_wt = result[:3]
    if return_ts or analyze:
        beams_ts = result[3]
    pix_flag_list = result[-1]

    data_file_header = build_header(data_header) + write_suffix
    if table_save:  # save to csv
        for obs, name in zip((beams_flux, beams_err),
                             ("beams_flux", "beams_err")):
            obs.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_%s.csv" % (data_file_header, name)),
                    overwrite=True)
        beams_flux.obs_info_.table_.write(os.path.join(
                write_dir, "%s_beams_info.csv" % data_file_header),
                overwrite=True)
    if plot:  # plot beam flux
        plot_dict = {"beam flux": [beams_flux, beams_err,
                                   {"c": "y", "ls": ":", "lw": 0.5}]}
        # plot PWV over beam flux
        if ("UTC" in beams_flux.obs_info_.table_.colnames) and \
                ("mm PWV" in beams_flux.obs_info_.table_.colnames):
            tb_use = beams_flux.obs_info_.table_[
                ~beams_flux.obs_info_.table_.mask["UTC"]]
            tb_use.sort("UTC")
            t_arr = Time.strptime(tb_use["UTC"],
                                  format_string="%Y-%m-%dU%H:%M:%S"). \
                to_value(format="unix")
            t_arr += tb_use["Scan duration"] / 2
            pwv_arr = tb_use["mm PWV"]
            obs_pwv = beams_flux.replace(
                    arr_in=np.tile(pwv_arr, beams_flux.shape_[:-1] + (1,)),
                    ts=t_arr, chop=None)
            plot_dict["PWV"] = [obs_pwv, {"ls": "--", "twin_axes": True,
                                          "c": "r", "marker": ".",
                                          "markersize": 3}]
        plt.close(plot_beam_ts(
                plot_dict, title=(data_file_header + " beam flux"),
                pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                plot_show=plot_show, plot_save=plot_save,
                write_header=os.path.join(
                        write_dir, "%s_beams_flux" % data_file_header),
                orientation=ORIENTATION))

    if analyze:
        beams_rms = analyze_performance(
                beams_ts, write_header=os.path.join(write_dir, data_file_header),
                pix_flag_list=pix_flag_list, plot=plot, plot_rms=plot_flux,
                plot_ts=False, reg_interest=reg_interest, plot_psd=plot_ts,
                plot_specgram=False, plot_show=plot_show, plot_save=plot_save)
        if table_save:
            beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_rms.csv" % data_file_header), overwrite=True)

    result = (beams_flux, beams_err, beams_wt)
    if return_ts:
        result += (beams_ts,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def reduce_zpold(data_header, data_dir=None, write_dir=None, write_suffix="",
                 array_map=None, obs_log=None, is_flat=False, pix_flag_list=[],
                 flat_flux=1, flat_err=0, parallel=False, do_desnake=False,
                 ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                 return_ts=False, return_pix_flag_list=True, table_save=True,
                 plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                 plot_show=False, plot_save=True, analyze=False,
                 nod=False, use_hk=True, zpold_shape=ZPOLD_SHAPE):
    """
    plot raster of zpold

    :param bool nod: bool, flag whether the zpold is nodding, if True, it means
        there should be
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_ica),
                            ("desnake", "smooth", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method
    data_file_header = build_header(data_header) + write_suffix
    if not nod:
        result = reduce_calibration(
                data_header=data_header, data_dir=data_dir, write_dir=write_dir,
                write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
                is_flat=is_flat, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
                flat_err=flat_err, parallel=parallel, do_desnake=do_desnake,
                ref_pix=ref_pix, do_smooth=do_smooth, do_ica=do_ica,
                spat_excl=spat_excl, return_ts=True, return_pix_flag_list=True,
                table_save=table_save, plot=plot, plot_ts=plot_ts,
                reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save, analyze=False)
    else:
        result = reduce_beam_pairs(
                data_header=data_header, data_dir=data_dir, write_dir=write_dir,
                write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
                is_flat=is_flat, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
                flat_err=flat_err, parallel=parallel, do_desnake=do_desnake,
                ref_pix=ref_pix, do_smooth=do_smooth, do_ica=do_ica,
                spat_excl=spat_excl, return_ts=True,
                return_pix_flag_list=True, plot=plot, plot_ts=plot_ts,
                reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save, use_hk=use_hk)
        beams_flux, beams_err, beams_wt, pix_flag_list = result[:3] + result[-1:]

        if table_save:  # save to csv
            for obs, name in zip((beams_flux, beams_err),
                                 ("beam_pairs_flux", "beam_pairs_err")):
                obs.to_table(orientation=ORIENTATION).write(os.path.join(
                        write_dir, "%s_%s.csv" % (data_file_header, name)),
                        overwrite=True)
            beams_flux.obs_info_.table_.write(os.path.join(
                    write_dir, "%s_beam_pairs_info.csv" % data_file_header),
                    overwrite=True)
        if plot:  # plot beam flux
            plot_dict = {"beam pair flux": [beams_flux, beams_err,
                                            {"c": "k", "ls": ":", "lw": 0.5}]}
            # plot PWV over beam flux
            if ("UTC" in beams_flux.obs_info_.table_.colnames) and \
                    ("mm PWV" in beams_flux.obs_info_.table_.colnames):
                tb_use = beams_flux.obs_info_.table_[
                    ~beams_flux.obs_info_.table_.mask["UTC"]]
                tb_use.sort("UTC")
                t_arr = Time.strptime(tb_use["UTC"],
                                      format_string="%Y-%m-%dU%H:%M:%S"). \
                    to_value(format="unix")
                t_arr += tb_use["Scan duration"] / 2
                pwv_arr = tb_use["mm PWV"]
                obs_pwv = beams_flux.replace(
                        arr_in=np.tile(pwv_arr, beams_flux.shape_[:-1] + (1,)),
                        ts=t_arr, chop=None)
                plot_dict["PWV"] = [obs_pwv, {"ls": "--", "twin_axes": True,
                                              "c": "r", "marker": ".",
                                              "markersize": 3}]
            plt.close(plot_beam_ts(
                    plot_dict, title=(data_file_header + " beam pair flux"),
                    pix_flag_list=pix_flag_list, reg_interest=reg_interest,
                    plot_show=plot_show, plot_save=plot_save,
                    write_header=os.path.join(
                            write_dir, "%s_beam_pairs_flux" % data_file_header),
                    orientation=ORIENTATION))
    beams_flux, beams_err, beams_wt, beams_ts, pix_flag_list = result

    raster_result = make_raster(
            beams_flux=beams_flux, beams_err=beams_err,
            write_header=os.path.join(write_dir, data_file_header),
            pix_flag_list=pix_flag_list, raster_shape=zpold_shape,
            return_pix_flag_list=True, plot=plot, reg_interest=reg_interest,
            plot_show=plot_show, plot_save=plot_save, raster_thre=RASTER_THRE)
    raster_flux, pix_flag_list = raster_result

    stacked_result = stack_raster(
            raster=raster_flux, raster_wt=(1 / (beams_err ** 2).
                                           proc_along_time(method="nanmean").sqrt()).data_[..., None],
            write_header=os.path.join(write_dir, data_file_header),
            pix_flag_list=pix_flag_list, plot=plot, plot_show=plot_show,
            plot_save=plot_save)

    if analyze:
        beams_rms = analyze_performance(
                result[3], write_header=os.path.join(
                        write_dir, data_file_header),
                pix_flag_list=result[-1], plot=plot, plot_rms=plot_flux,
                plot_ts=False, reg_interest=reg_interest, plot_psd=plot_ts,
                plot_specgram=False, plot_show=plot_show,
                plot_save=plot_save)
        if table_save:
            beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_rms.csv" % data_file_header),
                    overwrite=True)

    result = (raster_flux,)
    if return_ts:
        result += (beams_ts,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def reduce_zpoldbig(data_header, data_dir=None, write_dir=None, write_suffix="",
                    array_map=None, obs_log=None, is_flat=False, pix_flag_list=[],
                    flat_flux=1, flat_err=0, parallel=False, do_desnake=False,
                    ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                    return_ts=False, return_pix_flag_list=False, table_save=True,
                    plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                    plot_show=False, plot_save=True, analyze=False, nod=False,
                    use_hk=True, zpoldbig_shape=ZPOLDBIG_SHAPE):
    """
    raster shape according to zpoldbig
    """

    return reduce_zpold(
            data_header=data_header, data_dir=data_dir, write_dir=write_dir,
            write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
            is_flat=is_flat, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
            flat_err=flat_err, parallel=parallel, do_desnake=do_desnake,
            ref_pix=ref_pix, do_smooth=do_smooth, do_ica=do_ica,
            spat_excl=spat_excl, return_ts=return_ts,
            return_pix_flag_list=return_pix_flag_list, table_save=table_save,
            plot=plot, plot_ts=plot_ts, reg_interest=reg_interest,
            plot_flux=plot_flux, plot_show=plot_show, plot_save=plot_save,
            analyze=analyze, nod=nod, use_hk=use_hk, zpold_shape=zpoldbig_shape)


def eval_performance(data_header, data_dir=None, write_dir=None, write_suffix="",
                     array_map=None, obs_log=None, pix_flag_list=[],
                     parallel=False, return_ts=False, table_save=True,
                     plot=True, plot_ts=True, reg_interest=None, plot_psd=True,
                     plot_specgram=True, plot_flux=True, plot_show=False,
                     plot_save=True):
    """
    Read a batch of beams and run analyze_performance on the time series. Be
    cautious that plot_specgram can be very slow
    """

    if data_dir is None:
        data_dir = os.getcwd()
    if write_dir is None:
        write_dir = os.getcwd()
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix

    print("Reading data.")
    data_file_header = build_header(data_header) + write_suffix
    file_header_list = [os.path.join(data_dir, "%s_%04d" % (header, beam_num))
                        for header in data_header
                        for beam_ran in data_header[header]
                        for beam_num in range(beam_ran[0], beam_ran[1] + 1)]
    beams = read_beams(file_header_list, array_map=array_map if
    (array_map is None) or (reg_interest is None) else
    array_map.take_where(**reg_interest), obs_log=obs_log,
                       flag_ts=True, parallel=parallel)
    print("Analyzing data.")
    beams_rms = analyze_performance(
            beams, write_header=os.path.join(write_dir, data_file_header),
            pix_flag_list=pix_flag_list, plot=plot, plot_rms=plot_flux,
            plot_ts=plot_ts, reg_interest=reg_interest, plot_psd=plot_psd,
            plot_specgram=plot_specgram, plot_show=plot_show,
            plot_save=plot_save)
    if table_save:
        beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                write_dir, "%s_rms.csv" % data_file_header), overwrite=True)
        beams_rms.obs_info_.table_.write(os.path.join(
                write_dir, "%s_info.csv" % data_file_header), overwrite=True)

    result = (beams_rms,)
    if return_ts:
        result += (beams,)
    result += (pix_flag_list,)
    return result

# def raster_map
