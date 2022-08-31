"""
A submodule including many helper functions
"""

import warnings

import numpy as np
from numpy.linalg import lstsq


def custom_formatwarning(message, category, *args, **kwargs):
    # ignore everything except the message
    return "%s: %s\n" % (category.__name__, message)


warnings.formatwarning = custom_formatwarning


def gaussian(x, x0=0, sigma=1, amp=1, norm=False):
    """
    return the evaluation of gaussian distribution with center x0 and sigma at
    given x. If not normalized, the value at center is amp, otherwise the peak is
    amp/sqrt(2*np.pi*sigma)

    :param x: float or array, place to calculate the value of gaussian
        distribution
    :type x: float or numpy.ndarray
    :param float x0: float, center of gaussian distribution
    :param float sigma: float, standard deviation of gaussian
    :param float amp: float, amplitude of the gaussian peak; if norm=False, amp
        will be the peak value, otherwise amp will be the integrated value
    :param bool norm: bool, whether to normalize the peak to represent an
        actual pdf
    :return: array or value, evaluated at the given position in the same shape
        as input pos
    :rtype: float or numpy.ndarray
    """

    a = amp if not norm else amp / np.sqrt(2 * np.pi) / sigma

    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gaussian_2d(pos, x0=0, y0=0, sigma_x=1, sigma_y=1, theta=0, amp=1,
                norm=False):
    """
    return a 2d gaussian distribution

    :param pos: list or tuple or array, size two recording the values of x and y
        dimension to evaluate, can be (float, float) or (array, array) of the same
        size
    :type pos: list or tuple or numpy.ndarray
    :param float x0: float, center of gaussian distribution in x dimension
    :param float y0: float, center of gaussian distribution in y dimension
    :param float sigma_x: float, standard deviation of gaussian in the x dimension
    :param float sigma_y: float, standard deviation of gaussian in the y dimension
    :param float theta: float, radian of the position angle of x with respect to
        the first dimension
    :param float amp: float, amplitude of the gaussian peak; if norm=False, amp
        will be the peak value, otherwise amp will be the integrated value
    :param bool norm: bool, whether to normalize the peak to represent an
        actual pdf
    :return: array or value, evaluated at the given position in the same shape
        as input pos
    :rtype: float or numpy.ndarray
    """

    xx, yy = pos
    xx, yy = np.array(xx), np.array(yy)

    xxp = xx * np.cos(theta) + yy * np.sin(theta)
    x0p = x0 * np.cos(theta) + y0 * np.sin(theta)
    yyp = - xx * np.sin(theta) + yy * np.cos(theta)
    y0p = - x0 * np.sin(theta) + y0 * np.cos(theta)

    return amp * gaussian(x=xxp, x0=x0p, sigma=sigma_x, norm=norm) * \
           gaussian(x=yyp, x0=y0p, sigma=sigma_y, norm=norm)


def proc_array(arr, method="mean", axis=None, **kwargs):
    """
    a wrapper function to process array which enables to process array by
    specifying the function name by calling np.{method}(arr, axis=axis), also
    expands the range of available methods including 'nanmad', 'mad',
    'num', 'num_is_nan', 'num_not_is_nan', 'num_is_finite', 'num_not_is_finite'

    :param list or tuple or numpy.ndarray arr: array to be processed
    :param str method: str, method name to be used to process array, can either
        be a numpy function, or method names specified in the document
    :param int or None axis: int or None, axis to process the array, if left
        None, the function will be applied across the whole array
    :param dict kwargs: keyword arguments passed to np.{method}() function
    :return: processed array
    :rtype: numpy.ndarray
    :raises ValueError: invalid method name
    """

    func_dict = {"nanmad": lambda arr, axis: median_abs_deviation(
            arr, axis=axis, nan_policy="omit"),
                 "mad": lambda arr, axis: median_abs_deviation(
                         arr, axis=axis, nan_policy="propagate"),
                 "num": lambda arr, axis:
                 np.count_nonzero(np.ones(arr.shape), axis=axis),
                 "num_is_nan": lambda arr, axis:
                 np.count_nonzero(np.isnan(arr), axis=axis),
                 "num_not_is_nan": lambda arr, axis:
                 np.count_nonzero(~np.isnan(arr), axis=axis),
                 "num_is_finite": lambda arr, axis:
                 np.count_nonzero(np.isfinite(arr), axis=axis),
                 "num_not_is_finite": lambda arr, axis:
                 np.count_nonzero(~np.isfinite(arr), axis=axis)}

    if method in func_dict:
        func = func_dict[method]
    elif hasattr(np, method):
        func = np.__dict__[method]
    else:
        raise ValueError("Input method is not a numpy function or valid name.")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings(
                    "ignore", message="Degrees of freedom <= 0 for slice.")
            arr_proc = func(arr, axis=axis, **kwargs)
    except Exception as err:
        raise RuntimeError("Failed to call %s with %s." % (method, func)) \
            from err

    return arr_proc


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
    :rtype: tuple
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
        err = np.sqrt((((arr_use - mean) * wt_use) ** 2).sum()) / summed_wt
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
    :rtype: tuple
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
        if np.any(cumsum_wt == summed_wt / 2):
            mid_idx = np.flatnonzero(cumsum_wt == summed_wt / 2)[0]
            med = (arr_use[mid_idx] + arr_use[mid_idx + 1]) / 2
        else:
            mid_idx = np.flatnonzero(cumsum_wt > summed_wt / 2)[0]
            med = arr_use[mid_idx]
        err = np.sqrt((((arr_use - med) * wt_use) ** 2).sum()) / summed_wt
        return med, err, summed_wt


def index_diff_edge(arr, thre_min=0.5, thre_max=None):
    """
    Return the all edge index that splits the arr if the absolute difference of
    two neighbouring elements is higher than thre. The first elements is always
    0 and the last is the length of the input array, which is different from the
    input for numpy.split().

    :param arr: list or tuple or arr, 1 dimension of int or float or bool type
    :type arr: list or tuple or numpy.ndarray
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
    :return flag_arr: array, bool values of flag in the same shape as arr
    :rtype: numpy.ndarray
    :raises ValueError: invalid axis value
    """

    arr = np.asarray(arr)  # check input arr and axis
    ndim, shape = len(arr.shape), arr.shape
    if axis is not None:
        if int(axis) not in range(-ndim, ndim):  # check axis value
            raise ValueError("Invalid axis value: %i." % axis)
        axis = int(axis) if (axis >= 0) else int(arr.ndim + axis)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore",
                                message="invalid value encountered in greater")
        if (proc_array(arr, method="num_is_finite") == 0) and (axis is None):
            flag_arr = np.full(arr.shape, fill_value=False)
        else:
            med = np.nanmedian(arr, axis=axis, keepdims=True)
            abs_div = np.abs(arr - med)
            mad = np.nanmedian(abs_div, axis=axis, keepdims=True)
            flag_arr = (abs_div > mad * thre)

    return flag_arr


def double_nanmad_flag(arr, thre=20, axis=-1, frac_thre=0.1):
    """
    Similar to nanmad_flag(), but do nanmad_flag a second time on the data flagged
    by nanmad_flag(), if frac_thre of data is flagged in the first time in a
    certain entry of arr, to account for the sudden jump often witnessed in time
    series or bimodal distribution

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

    arr = np.asarray(arr)  # check input arr and axis
    flag_arr = nanmad_flag(arr=arr, thre=thre, axis=axis)
    flagged_arr = np.full_like(arr, fill_value=np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="invalid value encountered in greater")
        do_flag = (np.nansum(flag_arr, axis=axis, keepdims=True) /
                   flag_arr.shape[axis]) > frac_thre
    np.putmask(flagged_arr, flag_arr & do_flag, arr)
    second_flag_arr = nanmad_flag(arr=flagged_arr, thre=thre, axis=axis)
    np.putmask(flag_arr, np.full_like(flag_arr, fill_value=True) & do_flag,
               second_flag_arr)

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
    if np.count_nonzero(finite_mask) == 0:
        result = np.full(x.shape, fill_value=fill_value)
    else:
        result = np.interp(x, xp[finite_mask], fp[finite_mask])

    return result


def nanlstsq(a, b, rcond=None, fill_value=np.nan):
    """
    call numpy.linalg.lstsq() to do the least square fit, but mask non-finite data
    first, and return all the results from

    :param numpy.ndarray a: array, as in numpy.linalg.lstsq()
    :param numpy.ndarray b: array, 1-d, as in numpy.linalg.lstsq()
    :param rcond: float or str, as in numpy.linalg.lstsq()
    :param fill_value: value to fill the result array in case all elements in
        xp are non-finite
    """

    finite_flag_arr = np.all(np.isfinite(a), axis=-1) & np.isfinite(b)
    if np.count_nonzero(finite_flag_arr) == 0:
        lstsq_0 = lstsq(np.zeros(a.shape), np.zeros(b.shape), rcond=rcond)
        result = (np.full(lstsq_0[0].shape, fill_value=fill_value),) + lstsq_0[1:]
    else:
        result = lstsq(a[finite_flag_arr, :], b[finite_flag_arr], rcond=rcond)

    return result


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
