# @Date    : 2021-01-29 16:26:51
# @Credit  : Bo Peng(bp392@cornell.edu), Cody Lamarche, Christopher Rooney
# @Name    : tools.py
# @Version : beta
"""
A package that has many helper functions
"""

import warnings

import numpy as np
from numpy.linalg import lstsq


def gaussian(x, x0=0, sigma=1, norm=False):
    """
    return the evaluation of gaussian distribution with center x0 and sigma at
    given x. If not normalized, the value at center is 1, other wise the peak is
    1/sqrt(2*np.pi*sigma)

    :param x: float or array, place to calculate the value of gaussian
        distribution
    :type x: float or numpy.ndarray
    :param float x0: float, center of gaussian distribution
    :param float sigma: float, standard deviation of gaussian
    :param bool norm: bool, whether to normalized the peak to represent an
        actual pdf
    """

    a = 1 if not norm else 1/np.sqrt(2 * np.pi * sigma)

    return a * np.exp(-(x - x0)**2/(2 * sigma**2))


def weighted_mean(arr, wt=None, nan_policy="omit"):
    """
    Calculate the weighted mean of 1-d array, return the flux, error and summed
    weight. Supported nan_policy are 'omit'(equivalent to 'ignore') and
    'propagate'.

    :param numpy.ndarray arr: array to be averaged
    :param numpy.ndarray wt: array of weight of each data point, if left none
        will use array of 1
    :param str nan_policy: str flag whether to 'omit'(or 'ignore') nan value in
        the data, or to 'propagate' to the result, in which case the mean and
        error will be nan, and weight will be 0
    :return: tuple of (mean, error, summed_weight)
    :type: tuple
    :raises ValueError: inconsistent shape, invalid nan_policy value
    """

    arr = np.array(arr)
    if wt is None:
        wt = np.ones(arr.shape)
    else:
        wt = abs(np.array(wt))
        if arr.shape != wt.shape:
            raise ValueError("Inconsistent shape between arr and wt.")

    if nan_policy.strip().lower()[0] in ["o", "i"]:
        finite_mask = np.isfinite(arr) & np.isfinite(wt)
    elif nan_policy.strip().lower()[0] == "p":
        finite_flag = np.all(np.isfinite(arr)) and np.all(np.isfinite(wt))
        finite_mask = np.full(arr.shape, fill_value=finite_flag, dtype=bool)
    else:
        raise ValueError("Invalid value for nan_policy")

    if np.count_nonzero(finite_mask) == 0:
        return np.nan, np.nan, 0
    else:
        arr_use, wt_use = arr[finite_mask], wt[finite_mask]
        summed_wt = wt_use.sum()
        mean = (arr_use * wt_use).sum() / summed_wt
        err = np.sqrt((((arr_use - mean) * wt_use)**2).sum()) / summed_wt
        return mean, err, summed_wt


def weighted_median(arr, wt=None, nan_policy="omit"):
    """
    Calculate the weighted median of 1-d array, return the flux, error and
    summed weight. Supported nan_policy are 'omit'(equivalent to 'ignore') and
    'propagate'.

    :param numpy.ndarray arr: array containing data
    :param numpy.ndarray wt: array of weight of each data point, if left none
        will use array of 1
    :param str nan_policy: str flag whether to 'omit' nan value in the data,
        or to 'propagate' to the result, in which case the mean and error will
        be nan, and weight will be 0
    :return: tuple of (median, error, summed_weight)
    :type: tuple
    :raises ValueError: inconsistent shape, invalid nan_policy value
    """

    arr = np.array(arr)
    if wt is None:
        wt = np.ones(arr.shape)
    else:
        wt = np.array(wt)
        if arr.shape != wt.shape:
            raise ValueError("Inconsistent shape between arr and wt.")

    if nan_policy.strip().lower()[0] in ["o", "i"]:
        finite_mask = np.isfinite(arr) & np.isfinite(wt)
    elif nan_policy.strip().lower()[0] == "p":
        finite_flag = np.all(np.isfinite(arr)) and np.all(np.isfinite(wt))
        finite_mask = np.full(arr.shape, fill_value=finite_flag, dtype=bool)
    else:
        raise ValueError("Invalid value for nan_policy")

    if np.count_nonzero(finite_mask) == 0:
        return np.nan, np.nan, 0
    else:
        arr_use = arr[finite_mask]
        idxs = np.argsort(arr_use)
        arr_use, wt_use = arr_use[idxs], wt[finite_mask][idxs]
        summed_wt = wt_use.sum()
        cumsum_wt = np.cumsum(wt_use)
        if np.any(cumsum_wt == summed_wt/2):
            mid_idx = np.flatnonzero(cumsum_wt == summed_wt/2)[0]
            med = (arr_use[mid_idx] + arr_use[mid_idx + 1])/2
        else:
            mid_idx = np.flatnonzero(cumsum_wt > summed_wt/2)[0]
            med = arr_use[mid_idx]
        err = np.sqrt((((arr_use - med) * wt_use)**2).sum()) / summed_wt
        return med, err, summed_wt


def index_diff_edge(arr, thre_min=0.5, thre_max=None):
    """
    Return the all edge index that splits the arr if the absolute difference of
    two neighbouring elements is higher than thre. The first elements is always
    0 and the last is the length of the input array, which is different from the
    input for numpy.split().

    :param arr: list or tuple or arr, 1 dimension of int or float or bool type
    :type arr: Union[list, tuple, numpy.ndarray]
    :param float thre_min: float, lower threshold of absolute difference to cut
        the arr
    :param thre_max: float, optional upper threshold of absolute
        difference to cut the arr. Abs difference > thre_max will be ignored
    :type thre_max: float
    :return edge_idxs: array, index of edge element of each chunk
    :rtype: numpy.ndarray
    :example:
    >>> edge_idxs = index_diff_edge([0, 0, 0, 1, 1, 0, 1])

        will return array([0, 3, 5, 6, 7]), such that
        (edge_idxs[i]:edge_idxs[i+1])
        selects the part that are in the same chop phase

    >>> edge_idxs = index_diff_edge(
    >>> [0, 0.0025, 0.05, 0.3, 0.3025, 0.305, 0.3075], thre=0.1)

        will return array([0, 3, 7])

    >>> index_diff_edge([0])

        will return [0, 1]
    """

    abs_diff = abs(np.diff(arr))
    flag_edge = (abs_diff > thre_min) if thre_max is None else \
        ((abs_diff > thre_min) & (abs_diff < thre_max))
    edge_diff_thre = list(np.flatnonzero(flag_edge) + 1)
    edge_idxs = np.unique([0] + edge_diff_thre + [len(arr)])
    edge_idxs.sort()

    return edge_idxs


def median_abs_deviation(arr, axis=0, nan_policy='omit', keepdims=False):
    """
    Re-implementation of scipy.stat.median_abs_deviation()
    """

    if nan_policy.strip().lower()[0] == "p":
        func = np.median
    elif nan_policy.strip().lower()[0] in ["o", "i"]:
        func = np.nanmedian
    else:
        raise ValueError("Invalid value for nan_policy.")

    med = func(arr, axis=axis, keepdims=True)
    abs_div = np.abs(arr - med)
    mad = func(abs_div, axis=axis, keepdims=keepdims)

    return mad


def nanmad_flag(arr, thre=20, axis=-1):
    """
    Return a bool array in which the elements with distance to the median
    along the given axis larger than threshold times the median absolute
    deviation (MAD) are flagged as True, otherwise False.

    :param numpy.ndarray arr: array, to be checked
    :param float thre: float, data with abs distance > thre*MAD will be flagged
    :param int axis: int, axis along which the mad will be checked, if input is
        None, will use the median of the whole array
    :return flag_arr: array, bool values of flag
    :rtype: numpy.ndarray
    :raises ValueError: invalid axis value
    """

    arr = np.array(arr)  # check input arr and axis
    ndim, shape = len(arr.shape), arr.shape
    if axis is not None:
        if int(axis) not in range(-ndim, ndim):  # check axis value
            raise ValueError("Invalid axis value: %i." % axis)
        axis = int(axis) if (axis >= 0) else int(arr.ndim + axis)

    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered")
        med = np.nanmedian(arr, axis=axis, keepdims=True)
    abs_div = np.abs(arr - med)
    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="invalid value encountered in greater")
        warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered")
        mad = np.nanmedian(abs_div, axis=axis, keepdims=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="invalid value encountered in greater")
        warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered")
        flag_arr = (abs_div > mad * thre)

    return flag_arr


def double_nanmad_flag(arr, thre=20, axis=-1, frac_thre=0.1):
    """
    Similar to nanmad_flag(), but do nanmad_flag a second time on the data flagged
    by nanmad_flag(), if frac_thre of data is flagged in the first time in a
    certain entry of arr, to account for the sudden jump often witnessed in time
    series

    :param numpy.ndarray arr: array, to be checked
    :param float thre: float, data with abs distance > thre*MAD will be flagged
    :param int axis: int, axis along which the mad will be checked, if input is
        None, will use the median of the whole array
    :param flat frac_thre: float, between 0 and 1, threshold of the fraction of
        data flagged in the first time nanmad_flag() to perform nanmad_flag() and
        try to unflag some data
    :return flag_arr: array, bool values of flag
    :rtype: numpy.ndarray
    :raises ValueError: invalid value for frac_thre
    """

    if not 0 <= frac_thre <= 1:
        raise ValueError("Invalid value for frac_thre.")
    arr = np.array(arr)  # check input arr and axis
    first_flag_arr = nanmad_flag(arr=arr, thre=thre, axis=axis)
    flagged_arr, unflagged_arr = \
        np.full_like(arr, fill_value=np.nan, dtype=arr.dtype), \
        np.full_like(arr, fill_value=np.nan, dtype=arr.dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="invalid value encountered in greater")
        do_flag = (np.nansum(first_flag_arr, axis=axis, keepdims=True) /
                   first_flag_arr.shape[axis]) > frac_thre
    flagged_arr[first_flag_arr & do_flag] = arr[first_flag_arr & do_flag]
    unflagged_arr[~first_flag_arr & do_flag] = arr[~first_flag_arr & do_flag]
    second_flag_arr = nanmad_flag(flagged_arr) | nanmad_flag(unflagged_arr)
    flag_arr = first_flag_arr
    flag_arr[np.full_like(first_flag_arr, fill_value=True) & do_flag] = \
        second_flag_arr[np.full_like(first_flag_arr, fill_value=True) & do_flag]

    return flag_arr


def naninterp(x, xp, fp, fill_value=np.nan):
    """
    Call numpy.interp() to do a 1d interpolation, but mask the non-finite data
    first, and return an array with fill_value in case all input data are not
    finite

    :param numpy.ndarray x: array, as in numpy.interp()
    :param numpy.ndarray xp: array numpy.interp()
    :param numpy.ndarray fp: array, numpy.interp()
    :param fill_value: value to fill the result array in case all elements in
        xp are non-finite
    :return: array of interpolated result
    :rtype: numpy.ndarray
    """

    finite_mask = np.isfinite(xp) & np.isfinite(fp)
    if len(finite_mask) == 0:
        return np.full(x.shape, fill_value=fill_value)
    else:
        return np.interp(x, xp[finite_mask], fp[finite_mask])


def nanlstsq(a, b, rcond=None, fill_value=np.nan):
    """
    call numpy.linalg.lstsq() to do least square fit, but mask non-finite data 
    first, and return all the results from

    :param numpy.ndarray a: array, as in numpy.linalg.lstsq()
    :param numpy.ndarray b: array, 1-d, as in numpy.linalg.lstsq()
    :param rcond: float or str, as in numpy.linalg.lstsq()
    :param fill_value: value to fill the result array in case all elements in
        xp are non-finite
    """

    finite_flag_arr = np.all(np.isfinite(a), axis=-1) & np.isfinite(b)
    if len(finite_flag_arr) == 0:
        lstsq_0 = lstsq(np.zeros(a.shape), np.zeros(b.shape), rcond=rcond)
        return (np.full(lstsq_0[0].shape, fill_value=fill_value),) + lstsq_0[1:]
    else:
        return lstsq(a[finite_flag_arr, :], b[finite_flag_arr], rcond=rcond)


def check_orientation(orientation):
    """
    Check if orientation is valid input or not. Return True if is 'horizontal',
    False if 'vertical', raise Error if invalid input

    :param str orientation: str, allowed values are 'horizontal' and 'vertical'
    :return: True if is horizontal, False if vertical
    :rtype: bool
    :raises ValueError: invalid input
    """

    if orientation.lower().strip()[0] == "h":
        return True
    elif orientation.lower().strip()[0] == "v":
        return False
    else:
        raise ValueError("Invalid input orientation: %s." % orientation)


def build_header(header_dict):
    """
    format the file header string based on the input header dict

    :param dict header_dict: dict, in the format
        {'file_header1': [(start_beam1, end_beam_1), ...], 'file_header2':...}
    :return: str, 'file_header1_start_beam1-end_beamN' if there is only one
        file_header, otherwise file_header1_start_beam1-file_headerN_end_beamM'
    :rtype: str
    """

    if len(header_dict) == 1:
        header_str = "%s_%04d-%04d" % \
                (list(header_dict.items())[0][0],
                 list(header_dict.items())[0][1][0][0],
                 list(header_dict.items())[0][1][-1][-1])
    else:
        header_str = "%s_%04d-%s_%04d" % \
                        (list(header_dict.items())[0][0],
                         list(header_dict.items())[0][1][0][0],
                         list(header_dict.items())[-1][0],
                         list(header_dict.items())[-1][1][-1][-1])

    return header_str
