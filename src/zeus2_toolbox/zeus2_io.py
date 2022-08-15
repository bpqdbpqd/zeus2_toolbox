# @Date    : 2021-01-29 16:26:51
# @Credit  : Bo Peng(bp392@cornell.edu), Cody Lamarche, Christopher Rooney
# @Name    : zeus2_io.py
# @Version : 2.0
"""
A package that can help read in MCE data as well as ancillary data such as array
map, .chop file, .hk file, .ts file, obs_array log, etc. All the data will be
stored as objects that contain methods for data selection and simple reduction.
"""

import configparser
import copy
import gc
import inspect
import os.path
from collections import Counter
from datetime import datetime, timezone

import astropy
from astropy.table import vstack, hstack, unique, Table as Tb, Row
from astropy.time import Time

from .mce_data import *
from .tools import *


class BaseObj(object):
    """
    An object that can initialize from instance
    """

    empty_flag_ = True  # type: bool # whether the object is empty
    obj_type_ = "BaseObj"  # type: str
    len_ = 0  # type: int

    def __from_instance__(self, obj):
        if not isinstance(obj, type(self)):
            raise TypeError('Invalid input object type.')
        self.__dict__.update(obj.__dict__)

    def __repr__(self):
        return "%s, len=%i" % (self.obj_type_, self.len_)

    def __len__(self):
        return self.len_

    def copy(self):
        """
        Deep copy of the object.

        :return: a deep copy of the current object
        """
        return copy.deepcopy(self)


class TableObj(BaseObj):
    """
    An object hosting astropy.table.Table with some handy methods.
    Any child class of DataObj should accept the variable tb_in in __init__().
    """

    obj_type_ = "TableObj"  # type: str
    table_ = None  # type: astropy.table.Table # table containing .hk info
    colnames_ = []  # type: list[str, ]
    dtype_ = np.dtype([])  # type: numpy.dtype

    def __init__(self, tb_in=None):
        """
        Initialize TableObj object.

        :param tb_in: astropy.table class or TableObj instance
        :type tb_in: astropy.table.table.Table or TableObj
        """

        super(TableObj, self).__init__()
        if tb_in is None:
            tb_in = Tb(masked=True)

        if isinstance(tb_in, type(self)):
            self.__from_instance__(tb_in)
        else:
            if isinstance(tb_in, TableObj):
                tb_in = tb_in.table_.copy()
            elif isinstance(tb_in, Tb):
                if (len(tb_in) > 0) or (len(tb_in.colnames) > 0):
                    tb_in = Tb(tb_in, masked=True, copy=True)
                else:
                    tb_in = Tb(masked=True)
            elif isinstance(tb_in, np.ndarray):
                tb_in = Tb(tb_in, masked=True, copy=True)
            else:
                raise TypeError("Invalid type of tb_in. Expect astropy.table.")
            self.__fill_values__(tb_in=tb_in)

    def __fill_values__(self, tb_in):
        """
        Update table_ and other instances.

        :param astropy.table.table.Table tb_in: astropy.table, to use as
            table_
        """

        self.len_ = len(tb_in)
        self.table_ = tb_in
        self.colnames_ = tb_in.colnames
        self.dtype_ = tb_in.dtype
        self.empty_flag_ = True if (self.len_ == 0) and \
                                   (len(self.colnames_) == 0) else False

    def __eq__(self, other):
        """
        Compare table_ instances

        :param other: TableObj or child class, to compare with
        :type other: TableObj or child class
        :return same_flag: bool, flag whether the two objects are the same
        :rtype: bool
        :raises TypeError:
        """

        # check input type
        same_flag = True if isinstance(other, type(self)) else False
        if same_flag:  # only compare if there are the same type
            if self.empty_flag_ and other.empty_flag_:
                same_flag = same_flag & True
            else:
                same_flag = same_flag & (self.empty_flag_ ==
                                         other.empty_flag_)
                same_flag = same_flag & (self.len_ == other.len_)
                same_flag = same_flag & (self.dtype_ == other.dtype_)
                if same_flag:  # only check table_ if all other things agree
                    same_flag & np.all(self.table_ == other.table_)

        return same_flag

    def append(self, other):
        """
        Use vstack_reconcile() to append another TableObj object to the end
            of the current one

        :param other: HKInfo, object to append
        :type other: DataObj or child class
        :raises TypeError: invalid input type of hk_other
        """

        if not isinstance(other, TableObj):
            raise TypeError("Invalid input type for other.")
        if other.empty_flag_:
            pass
        else:
            if self.empty_flag_:
                table_new = other.table_
            else:
                table_new = vstack_reconcile([self.table_, other.table_],
                                             join_type="outer")
            self.__fill_values__(tb_in=table_new)

    def expand(self, other):
        """
        Use astropy.table.hstack() to add another TableObj object to the
        right side of the current one

        :param other: HKInfo, object to append
        :type other: DataObj or child class
        :raises TypeError: invalid input type of hk_other
        """

        if not isinstance(other, TableObj):
            raise TypeError("Invalid input type for other.")
        if other.empty_flag_:
            pass
        else:
            if self.empty_flag_:
                table_new = other.table_
            else:
                table_new = hstack([self.table_, other.table_],
                                   join_type="outer")
            self.__fill_values__(tb_in=table_new)

    def add_id(self, obs_id=0):
        """
        Add a column named 'id' in the table filled with the specified obs_id
        value. This can be handy to distinguish entries of different observation
        when observations are combined

        :param str obs_id: str, value to fill in the 'id' column
        """

        if not self.empty_flag_:
            try:
                self.table_.add_column(Tb.Column([obs_id] * self.len_,
                                                 name="obs_id", dtype="<U40"),
                                       index=0)
            except ValueError:
                pass

    def take_by_flag(self, flag_arr):
        """
        Create a new object from the current TableObj whose rows are selected by
        the bool array flag_arr

        :param flag_arr: bool array, 1-d, the length should match with
            self.len_, the rows of table_ flagged as True will be used in the
            new TableObj
        :type flag_arr: list or tuple or numpy.ndarray
        :return table_new: TableObj, a new object containing the data flagged
        :rtype: TableObj or child class
        :raise ValueError: inconsistent length
        """

        flag_arr = np.array(flag_arr, dtype=bool)
        if (len(flag_arr.shape) > 1) or (self.len_ != len(flag_arr)):
            raise ValueError("Inconsistent length of input flag_arr.")
        table_cut = self.table_[flag_arr]
        table_new = self.__class__(tb_in=table_cut)

        return table_new


class DataObj(BaseObj):
    """
    a general object that can host data array and chunk information of data. The
    data can be ndim = 1, 2, 3, 4, etc array, and the last axis is always
    considered as the axis of time, which is also the len of this object; the
    first axis is the row of pixel, and the second axis is the column.
    Any child class of DataObj should accept the variable arr_in in __init__().
    """

    obj_type_ = "DataObj"  # type: str
    data_ = np.empty(0, dtype=float)  # type: numpy.ndarray
    # instance variable hosting data array
    dtype_ = None  # type: numpy.dtype or None # type of data in array
    shape_ = (0,)  # type: tuple # shape of data
    ndim_ = 0  # type: int # number of dimension of data

    def __init__(self, arr_in=None):
        """
        Create an DataObj based on the type of input data, and assign len_,
        ndim_, dtype_, shape_, data_ etc. according to the input arr_in.
        If input arr_in is 1-d zero length array like np.array([]), then an
        empty object will be returned which can be appended by or append to any
        other DataObj.
        If input arr_in is empty but has dimension > 1, then the DataObj will be
        of the same shape, and can only append object of the same shape in all
        but the last axis.

        :param arr_in: array or DataObj object, containing data
        :type arr_in: numpy.ndarray or DataObj or list or tuple
        :raises TypeError: invalid input type
        """

        super(DataObj, self).__init__()
        if arr_in is None:
            arr_in = np.empty(0, dtype=float)

        if isinstance(arr_in, type(self)) and \
                isinstance(self, type(arr_in)):
            self.__from_instance__(arr_in)
        else:
            if isinstance(arr_in, DataObj):
                arr_in = arr_in.data_
            elif isinstance(arr_in, (np.ndarray, list, tuple)):
                arr_in = np.asarray(arr_in)
            else:
                raise TypeError("Invalid type of arr_in. Expect array.")
            self.__fill_values__(arr_in=arr_in)

    def replace(self, **kwargs):
        """
        Return a new DataObj with the input parameters replaced but copy all
        other instance variables.

        :param kwargs: key word arguments to initialize an object, for
            compatibility with subclasses
        :return: new object
        :rtype: DataObj
        """

        if "arr_in" not in kwargs:
            kwargs["arr_in"] = self.__dict__["data_"]

        return self.__class__(**kwargs)

    def __fill_values__(self, arr_in):
        """
        Assign values to instance variables using input array

        :param numpy.ndarray arr_in: array, containing data
        """

        arr_use = np.asarray(arr_in)
        if (arr_use.shape[-1] == 0) and (len(arr_use.shape) == 1):
            self.len_ = 0
            self.shape_ = (0,)
            self.ndim_ = 1
            self.dtype_ = None
            self.data_ = np.empty(self.shape_, dtype=self.dtype_)
            self.empty_flag_ = True
        else:
            self.len_ = arr_use.shape[-1]
            self.ndim_ = arr_use.ndim
            self.shape_ = arr_use.shape
            if self.dtype_ is None:
                self.dtype_ = arr_use.dtype
            self.data_ = arr_use.astype(self.dtype_, copy=True)
            self.empty_flag_ = False

    def __repr__(self):
        return super(DataObj, self).__repr__() + " shape=%s dtype=%s" % \
               (str(self.shape_), str(self.dtype_))

    def __operate__(self, other, operator, r=False):
        """
        Operate on the values in the data_ with another object or array or
        value, return a new DataObj with the new data and origin_.
        If one of them is empty, the other will be copied and returned.
        The other object or array can have different shape if either 1) both
        have the same ndim_, one has only size-one in one of the axis but the
        other has >1 size in that axis; e.g. (5, 1) matches (5, 7), (3, 1, 4)
        matches (3, 5, 4); 2)ndim_ of one object is smaller than the other by
        one, and the former has the shape_matches all but one value of the shape
        of the other object in the same order, e.g. shape (5,) matches (5, 7),
        (3, 4) matches (3, 5, 4). In the first case, the value of the size-one
        non-match axis will be operated with all the values in that axis for the
        other object, in the second case, the non-match axis will be treated as
        size 1, e.g. (5,) will be regarded as (5, 1) to operate with shape
        (5, 7) object. In the situation of ambiguity like matching (2, 3) to
        (2, 3, 3), the extra axis is always added backwards, such that (2, 3) is
        treated as (2, 3, 1).
        The output object dtype will be the dtype of the corresponding array
        operation result dtype

        :param other: DataObj or list or tuple or array or value, to be
            operated with the current data object
        :type other: DataObj or list or tuple or numpy.ndarray or int or
            float or np.double or bool
        :param operator: function, numpy function that operates on array
        :type operator: function or numpy.ufunc
        :param bool r: bool, default False, reverse operation flag
        :return: DataObj, with data_ operated
        :rtype: DataObj or subclass
        :raises TypeError: invalid type
        :raises ValueError: can not operate between shapes
        """

        if isinstance(other, (int, float, np.double, bool)):  # other is value
            if self.empty_flag_ or (self.len_ == 0):
                return self.copy()
            else:
                arr_new = operator(self.data_, other) if not r else \
                    operator(other, self.data_)
                return self.replace(arr_in=arr_new)
        if not isinstance(other, type(self)):  # not object
            if isinstance(other, (numpy.ndarray, list, tuple, DataObj)):
                other = DataObj(other)  # array to object
            else:
                raise TypeError("Invalid type, can only be operated on " +
                                "%s, list, tuple, array or value."
                                % str(type(self)))

        if self.empty_flag_:  # current object is empty
            return self.__class__(arr_in=other)
        if other.empty_flag_:  # data other is empty
            return self.copy()

        shape1, shape2 = self.shape_, other.shape_
        arr1, arr2 = self.data_, other.data_
        shape_flag = (shape1 == shape2)
        if shape_flag:  # the shapes match
            pass
        elif self.ndim_ == other.ndim_:  # ndim match, will try to reshape
            shape_flag = True
            for i in range(self.ndim_):
                if (shape1[i] != 1) and (shape2[i] != 1) and \
                        (shape1[i] != shape2[i]):
                    shape_flag = False
        else:
            axis_new = None
            if (self.ndim_ - other.ndim_) == 1:  # other ndim smaller
                for i in reversed(range(self.ndim_)):
                    if (shape1[:i] == shape2[:i]) and \
                            (shape1[i + 1:] == shape2[i:]):
                        shape_flag, axis_new = True, i
                        break
                if shape_flag and (axis_new is not None):
                    arr2 = np.expand_dims(arr2, axis=axis_new)
            elif (self.ndim_ - other.ndim_) == -1:  # self ndim smaller
                for i in reversed(range(other.ndim_)):
                    if (shape2[:i] == shape1[:i]) and \
                            (shape2[i + 1:] == shape1[i:]):
                        shape_flag, axis_new = True, i
                        break
                if shape_flag and (axis_new is not None):
                    arr1 = np.expand_dims(arr1, axis=axis_new)

        if not shape_flag:  # can not get shapes to match
            raise ValueError("Can not operate between shape %s and %s." %
                             (str(shape1), str(shape2)))
        else:
            arr_new = operator(arr1, arr2) if not r else operator(arr2, arr1)
            if (self.ndim_ - other.ndim_) == -1:
                return other.replace(arr_in=arr_new)
            else:
                return self.replace(arr_in=arr_new)

    def __add__(self, other):
        """
        Add the values in the data_ with another object or array or value,
        return a new DataObj with the new data and origin_.
        If one of them is empty, the other will be copied and returned.
        For the requirement of the shape of input, see the docstring for
        __operate() method.
        If the dtype does not match, the returned DataObj will use np.double

        :param other: DataObj or list or tuple or array or value, to be
            added with the current data object
        :type other: DataObj or list or tuple or numpy.ndarray or int or
            float or np.double or bool
        :return: DataObj, with added data
        :rtype: DataObj
        """
        return self.__operate__(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__operate__(other, np.subtract)

    def __rsub__(self, other):
        return self.__operate__(other, np.subtract, r=True)

    def __mul__(self, other):
        return self.__operate__(other, np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in true_divide")
            warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in true_divide")
            return self.__operate__(other, np.divide)

    def __rtruediv__(self, other):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in true_divide")
            warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in true_divide")
            return self.__operate__(other, np.divide, r=True)

    def __floordiv__(self, other):
        return self.__operate__(other, np.floor_divide)

    def __mod__(self, other):
        return self.__operate__(other, np.remainder)

    def __lt__(self, other):
        return self.__operate__(other, np.less)

    def __le__(self, other):
        return self.__operate__(other, np.less_equal)

    def __gt__(self, other):
        return self.__operate__(other, np.greater)

    def __ge__(self, other):
        return self.__operate__(other, np.greater_equal)

    def __and__(self, other):
        return self.__operate__(other, np.logical_and)

    def __or__(self, other):
        return self.__operate__(other, np.logical_or)

    def __xor__(self, other):
        return self.__operate__(other, np.logical_xor)

    def __not__(self):
        arr_new = np.logical_not(self.data_)
        return self.replace(arr_in=arr_new)

    def __invert__(self):
        arr_new = np.invert(self.data_)
        return self.replace(arr_in=arr_new)

    def __neg__(self):
        return self.__rsub__(0)

    def __pow__(self, power):
        return self.__operate__(power, np.power)

    def __rpow__(self, other):
        return self.__operate__(other, np.power, r=True)

    def __abs__(self):
        """
        Take the sqrt of the data_

        :return: DataObj or subclass, a new object with the same instance
            variables but data_ squared
        :rtype: DataObj or subclass(DataObj)
        """

        arr_new = abs(self.data_)
        return self.replace(arr_in=arr_new)

    def __eq__(self, other):
        """
        Compare the empty_flag_, dtype_, shape_, data_, with another DataObj
        object, return True if all instances are exactly the same values. Does
        not work if data_ contains np.nan, but it is okay with np.inf

        :param DataObj other: DataObj, the data object to compare with
        :return id_flag: flag whether the current chop is identical to the input
            chop
        :rtype: bool
        """

        # check input type
        same_flag = True if (isinstance(other, type(self)) &
                             isinstance(self, type(other))) else False
        if same_flag:  # only compare if they are the same type
            if self.empty_flag_ and other.empty_flag_:  # both are empty
                same_flag = same_flag & True
            else:
                if ((self.dtype_ in [float, np.double]) and
                    np.any(np.isnan(self.data_))) or \
                        ((other.dtype_ in [float, np.double]) and
                         np.any(np.isnan(other.data_))):
                    warnings.warn("DataObj contains np.nan, do not compare.",
                                  UserWarning)
                same_flag = same_flag & (self.empty_flag_ ==
                                         other.empty_flag_)
                same_flag = same_flag & (self.dtype_ == other.dtype_)
                same_flag = same_flag & (self.shape_ == other.shape_)
                same_flag = same_flag & np.all(self.data_ == other.data_)

        return same_flag

    def __ne__(self, other):
        return not self.__eq__(other)

    def sqrt(self):
        """
        Take the sqrt of the data_

        :return: DataObj or subclass, a new object with the same instance
            variables but data_ squared
        :rtype: DataObj or subclass(DataObj)
        """

        arr_new = np.sqrt(self.data_)
        return self.replace(arr_in=arr_new)

    def log(self):
        """
        Take log of e of the data_

        :return: DataObj or subclass, a new object with the same instance
            variables but data_ squared
        :rtype: DataObj or subclass(DataObj)
        """

        arr_new = np.log(self.data_)
        return self.replace(arr_in=arr_new)

    def log10(self):
        """
        Take log10 of the data_

        :return: DataObj or subclass, a new object with the same instance
            variables but data_ squared
        :rtype: DataObj or subclass(DataObj)
        """

        arr_new = np.log10(self.data_)
        return self.replace(arr_in=arr_new)

    def __check_axis__(self, axis):
        if axis is None:
            return axis
        else:
            if int(axis) not in range(-self.ndim_, self.ndim_):
                raise ValueError("Invalid axis value: %i." % axis)
            return int(axis) if (axis >= 0) else int(self.ndim_ + axis)

    def update_type(self, dtype):
        """
        Update dtype_ instance variable and dtype of data_ to the given value.
        Will not make change if object is empty.

        :param dtype: type information supported by numpy.dtype()
        :type dtype: type or numpy.dtype or str
        """

        self.dtype_ = np.dtype(dtype)
        self.__fill_values__(self.data_.astype(dtype))

    def as_type(self, dtype):
        """
        return an object with the data_ converted to the input dtype

        :param dtype: type information supported by numpy.dtype()
        :type dtype: type or numpy.dtype or str
        :return: new DataObj or subclass object with the data converted
        :rtype: DataObj or subclass
        """

        new_dtype = np.dtype(dtype)
        new_arr = self.data_.astype(new_dtype)

        return self.replace(arr_in=new_arr)

    def fill_by_mask(self, mask, fill_value=np.nan):
        """
        Change the elements that are flagged True in the mask to the specified
        value. Can be used to null the flagged elements, or change values of
        some elements.

        :param mask: array or DataObj or subclass object, shape is the same as
            data obj, with the elements to change flagged as True
        :type mask: numpy.ndarray or DataObj
        :param fill_value: default np.nan, value to fill. The value may not work
            as expected due to the dtype, e.g. bool(np.nan)
        :raises ValueError: invalid input shape
        """

        if isinstance(mask, DataObj):
            mask = mask.data_
        arr_new = self.data_
        np.putmask(arr_new, mask=mask, values=fill_value)
        self.__fill_values__(arr_new)

    def fill_by_flag_along_axis(self, flag_arr, fill_value=np.nan, axis=-1):
        """
        Change the elements that are flagged True in the flag_arr in the
        specified axis to the specified value. Can be used to null the flagged
        some pixels etc.

        :param numpy.ndarray flag_arr: bool array, 1-d, the length should match
            with shape_[axis], the data flagged as True will be changed
        :param fill_value: default np.nan, value to fill. The value may not work
            as expected due to the dtype, e.g. bool(np.nan)
        :param int axis: int, default -1,  axis index to apply the flag, allowed
            range is -ndim_ to ndim_-1
        :raises ValueError: invalid input shape
        :raise ValueError: invalid axis, inconsistent length
        """

        flag_arr = np.asarray(flag_arr, dtype=bool).flatten()
        axis = self.__check_axis__(axis=axis)
        if (len(flag_arr.shape) > 1) or (self.shape_[axis] != len(flag_arr)):
            raise ValueError("Inconsistent length of input flag_arr.")
        data_swap = self.data_.swapaxes(0, axis)
        data_swap[flag_arr] = fill_value
        self.data_ = data_swap.swapaxes(0, axis)

    def get_nanmad_flag(self, thre=10, axis=-1):
        """
        Return a bool array in which the elements with distance to the median
        along the given axis larger than threshold times the median absolute
        deviation (MAD) are flagged as True. Use nanmad_flag() function.

        :param float thre: float, data with abs distance > thre*MAD will be
            flagged
        :param int or None axis: int, axis along which the mad will be checked,
            if input is None, the data of the whole array will be used
        :return: array, bool values of flag
        :rtype: numpy.ndarray
        """

        if axis is not None:
            axis = self.__check_axis__(axis=axis)
        return nanmad_flag(self.data_, thre=thre, axis=axis)

    def get_double_nanmad_flag(self, thre=10, axis=-1, frac_thre=0.1):
        """
        similar to get_nanmad_flag(), but call double_nanmad_flag() to perform
        mad flagging twice, which is more robust for the sudden jump in time
        series and saves more data

        :param float thre: float, data with abs distance > thre*MAD will be flagged
        :param int or None axis: int, axis along which the mad will be checked,
            if input is None, will use the median of the whole array
        :param flat frac_thre: float, between 0 and 1, threshold of the fraction
            of data flagged in the first time nanmad_flag() to perform nanmad_flag()
            and try to unflag some data
        :return: array, bool values of flag
        :rtype: numpy.ndarray
        """

        if axis is not None:
            axis = self.__check_axis__(axis=axis)
        return double_nanmad_flag(self.data_, thre=thre, axis=axis,
                                  frac_thre=frac_thre)

    def take_by_flag_along_axis(self, flag_arr, axis=-1, **kwargs):
        """
        Create a new object from the current DataObj whose data are selected by
        the bool array flag in the axis specified.

        :param flag_arr: bool array, 1-d, the length should match with
            shape_[axis], the data flagged as True will be used in the new
            DataObj
        :type flag_arr: list or tuple or numpy.ndarray
        :param int axis: int, axis index to apply the flag, allow range is -ndim_
            to ndim_-1
        :param kwargs: keyword arguments to initialize a new object, for
            backward compatibility
        :return data_new: DataObj, a new object containing the data flagged
        :rtype: DataObj or child class
        :raise ValueError: invalid axis, inconsistent length
        """

        flag_arr = np.asarray(flag_arr, dtype=bool).flatten()
        axis = self.__check_axis__(axis=axis)
        if self.shape_[axis] != len(flag_arr):
            raise ValueError("Inconsistent length of input flag_arr.")
        arr_cut = np.compress(condition=flag_arr, a=self.data_, axis=axis)
        data_new = self.replace(arr_in=arr_cut, **kwargs)

        return data_new

    def take_by_idx_along_axis(self, idxs, axis=-1, **kwargs):
        """
        Create a new object from the current DataObj whose data are selected by
        the indices in idxs. Call take_by_flag_along_axis(), so the returned
        object may not have the same order as in idxs.

        :param idxs: int or list or tuple or numpy.ndarray, value or list of
            value of indices to take
        :type idxs: int or list or tuple or numpy.ndarray
        :param int axis: int, axis index to apply the flag, allow range is -ndim_
            to ndim_-1
        :param kwargs: keyword arguments to initialize a new object, for
            backward compatibility
        :return data_new: DataObj, a new object containing the data flagged
        :rtype: DataObj or child class
        :raise ValueError: invalid axis, inconsistent length
        """

        axis = self.__check_axis__(axis=axis)
        flag_arr = np.full(self.shape_[axis], fill_value=False, dtype=bool)
        flag_arr[idxs] = True

        return self.take_by_flag_along_axis(flag_arr=flag_arr, axis=axis,
                                            **kwargs)

    def append_along_axis(self, other, axis=-1):
        """
        Append another DataObj into the current chop in the given axis of data
        (the axis of time), and update the data_ instance of the current data.
        The shape of the two objects other than the specified axis must
        match to append, unless one of them is empty object with len_=0
        and ndim_=1. If the dtype_ does not match, then the data to append will
        be converted to the current dtype.

        :param DataObj other: DataObj, the data to append to the current
            DataObj object
        :param int axis: int, axis index to apply the flag, allow range is
            -ndim_ to ndim_-1
        :raises TypeError: wrong type of input, wrong ndim_ or shape_
        :raises ValueError: invalid axis
        """

        if not isinstance(other, DataObj):  # check type
            raise TypeError("Invalid type of input, expect DataObj instance.")
        axis = self.__check_axis__(axis=axis)
        if not (0 <= axis < self.ndim_):
            raise ValueError("Invalid value of axis.")

        if self.empty_flag_:  # the current object is empty
            self.__fill_values__(other.data_)
        elif other.empty_flag_:  # the object to append is empty
            pass
        else:  # both are npn-empty
            if (self.ndim_ != other.ndim_) or \
                    (self.shape_[:axis] != other.shape_[:axis]) or \
                    (self.shape_[axis + 1:] != other.shape_[axis + 1:]):
                raise ValueError("Incompatible shape_ %s and %s." %
                                 (str(self.shape_), str(other.shape_)))
            arr_new = np.concatenate((self.data_, other.data_), axis=axis)
            self.__fill_values__(arr_new)

    def proc_along_axis(self, method="nanmean", axis=-1, **kwargs):
        """
        Process data_ using the method in the specified axis. returns a new
        Object with the same ndim_, but length 1 in the given axis.

        :param str method: str, default 'nanmean', the function name to
            process data along the given axis by calling proc_array(); valid
            method names are numpy function names, and function names defined
            in proc_array() including 'nanmad', 'mad', 'num', 'num_is_nan',
            'num_not_is_nan', 'num_is_finite', 'num_not_is_finite'
        :param int axis: int, axis index to process the data, allow range is
            -ndim_ to ndim_-1
        :return data_proc: DataObj, dtype_ will be np.double for calculations
            or int for counting numbers, contained chunked data
        :param kwargs: keyword arguments of other parameters to replace in the
            returned object
        :rtype: DataObj
        :raise ValueError: invalid axis
        """

        axis = self.__check_axis__(axis=axis)
        dtype = int if method in ["num", "num_not_is_nan", "num_is_finite"] \
            else np.double

        arr_proc = proc_array(self.data_, method=method, axis=axis)
        arr_proc = np.expand_dims(arr_proc, axis=axis)
        arr_proc = arr_proc.astype(dtype)
        data_proc = self.replace(arr_in=arr_proc, **kwargs)

        return data_proc

    def get_index_chunk_edge(self, thre_min=0.5, thre_max=None):
        """
        If data is 1-d array, use index_diff_edge() to find the edge indices
        that can cut data into chunk based on the threshold of the absolute
        difference of neighbouring elements.

        :param float thre_min: float, default 0.5, threshold of absolute
            difference of neighbouring elements to chunk data, passed to
            index_diff_edge() function
        :param thre_max: float, optional upper threshold of absolute
            difference to cut the arr. Abs difference > thre_max will be ignored
        :type thre_max: float or None, optional
        :return chunk_edge_idxs: array, index of edge element of each chunk
        :rtype: numpy.ndarray
        :raises ValueError: wrong ndim
        """

        if self.ndim_ > 1:
            raise ValueError("Invalid ndim_. Expect 1.")
        chunk_edge_idxs = index_diff_edge(self.data_, thre_min=thre_min,
                                          thre_max=thre_max)

        return chunk_edge_idxs

    def chunk_split(self, chunk_edge_idxs):
        """
        Chunk the current data in the last axis according to the edge indices in
        chunk_edge_idxs, put each chunk into a new DataObj, return the list of
        the DataObjs.

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func
        :return chunk_list: list, containing DataObj objects of data chunks
            split by chunk_edge_idxs
        :rtype: list
        """

        if (chunk_edge_idxs[0] != 0) or (chunk_edge_idxs[-1] != self.len_):
            warnings.warn("Probably incomplete format for chunk_edge_idxs.")
        chunk_list = []
        for i, (idx_i, idx_e) in enumerate(zip(chunk_edge_idxs[:-1],
                                               chunk_edge_idxs[1:])):
            chunk_arr = self.data_[..., idx_i:idx_e]
            chunk_data = self.__class__(arr_in=chunk_arr)
            chunk_list.append(chunk_data)

        return chunk_list

    def chunk_proc(self, chunk_edge_idxs, method="nanmean", keep_shape=False,
                   **kwargs):
        """
        Chunk the current data in the last axis according to the edge indices in
        chunk_edge_idxs, and process data by mean/median/sum etc. in each chunk,
        and return a new DateObj object with processed chunk data.

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func
        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'nanmad'
            'mean', 'median', 'sum', 'std', 'min', 'max', 'mad',
            'num', 'num_is_nan', 'num_not_is_nan', 'num_is_finite',
            'num_not_is_finite', the method to calculate data value in each chunk
        :param keep_shape: bool, flag whether the output should have the shape
            as the input, so that each chunk are not compressed
        :param kwargs: keyword arguments of other parameters to replace in the
            object returned
        :return data_chunked: DataObj, dtype_ will be np.double for calculations
            or int for counting numbers, contained chunked data
        :rtype: DataObj
        :raises ValueError: invalid method value
        """

        dtype = int if method in ["num", "num_is_nan", "num_not_is_nan",
                                  "num_is_finite", "num_not_is_finite"] \
            else np.double
        if (chunk_edge_idxs[0] != 0) or (chunk_edge_idxs[-1] != self.len_):
            warnings.warn("Probably incomplete chunk_edge_idxs format.")

        chunk_num = len(chunk_edge_idxs) - 1
        chunk_shape = self.shape_ if keep_shape else \
            self.shape_[:-1] + (chunk_num,)
        chunk_arr = np.empty(chunk_shape, dtype=dtype)
        for i, (idx_i, idx_e) in enumerate(zip(chunk_edge_idxs[:-1],
                                               chunk_edge_idxs[1:])):
            arr_proc = proc_array(self.data_[..., idx_i:idx_e],
                                  method=method, axis=-1)
            if keep_shape:
                chunk_arr[..., idx_i:idx_e] = arr_proc[..., None]
            else:
                chunk_arr[..., i] = arr_proc

        data_chunked = self.replace(arr_in=chunk_arr, **kwargs)

        return data_chunked


class ArrayMap(DataObj):
    """
    mapping between MCE data and TES array. Can be initialized by a shape (m,n)
    array or list with n>=4, or use read() class method to read from .csv file.
    If using an array as the input to initialize, the [:, :4] will be passed to
    array_map_ instance variable
    """

    obj_type_ = "ArrayMap"  # type: str
    # dtype_ = np.dtype(int)  # type: numpy.dtype
    band_flag_ = False  # type: bool # flag if band is set
    band_ = 0  # type: int # band of the array map, optional
    array_map_ = np.empty((0, 4), dtype=int)  # type: numpy.ndarray
    # The second axis of the array are
    #   [:,0]: pixel spatial position(spat),
    #   [:,1]: spectral index(spec),
    #   [:,2]: MCE row,
    #   [:,3]: MCE column(col)
    pix_idxs_ = np.empty(0, dtype=int)  # type: numpy.ndarray
    # indices of pixels in the array map
    array_idxs_ = np.empty((0, 2), dtype=int)  # type: numpy.ndarray
    # TES array, (spat, spec)
    array_spat_ = np.empty(0, dtype=int)  # type: numpy.ndarray
    # TES array spatial position
    array_spec_ = np.empty(0, dtype=int)  # type: numpy.ndarray
    # TES array spectral index
    mce_idxs_ = np.empty((0, 2), dtype=int)  # type: numpy.ndarray
    # MCE index, (row, col)
    mce_row_ = np.empty(0, dtype=int)  # type: numpy.ndarray  # MCE row
    mce_col_ = np.empty(0, dtype=int)  # type: numpy.ndarray  # MCE column
    array_spat_llim_ = array_spat_ulim_ = -1  # range of array spat, default -1
    array_spec_llim_ = array_spec_ulim_ = -1
    mce_row_llim_ = mce_row_ulim_ = -1
    mce_col_llim_ = mce_col_ulim_ = -1
    wl_flag_ = False  # type: bool # flag whether wavelength is initialized
    conf_kwargs_ = dict()  # type: dict # keyword arguments of the grating and
    # telescope configuration, initialized by read_conf() and used in
    # init_wl() method
    array_wl_ = np.empty(0, dtype=float)  # type: numpy.ndarray # wavelength of
    # each pixel
    array_d_wl_ = np.empty(0, dtype=float)  # type: numpy.ndarray # wavelength

    # interval covered by each pixel

    @classmethod
    def read(cls, filename):
        """
        Return an ArrayMap object read from .csv file.

        :param str filename: str, path to the CSV file
        :return array_map: ArrayMap, new object
        :rtype: ArrayMap

        :example:
        array_map = ArrayMap.read(filename='array400_map.dat')
        """

        with open(filename, "rb") as file:  # Open the text file
            arr_in = np.genfromtxt(file, delimiter=",", comments="#",
                                   usecols=range(0, 7), dtype=int)
        array_map = cls(arr_in=arr_in)

        return array_map

    def __fill_values__(self, arr_in):
        """
        Initialize instance variables.

        :param numpy.ndarray arr_in: array, containing array map
        :return: None
        :rtype: NoneType
        raises: in valid dimension
        """

        super(ArrayMap, self).__fill_values__(arr_in=arr_in)
        if self.empty_flag_:
            self.array_map_ = np.empty((0, 4), dtype=self.dtype_)
        else:
            if self.ndim_ != 2:  # check shape of the input array
                raise ValueError("Invalid dimension for input array.")
            if self.shape_[-1] < 4:
                raise ValueError("Invalid last dimension for input array.")
            self.array_map_ = self.data_[:, (0, 1, 2, 3)]

        self.len_ = self.shape_[0]
        self.pix_idxs_ = np.arange(self.len_)

        self.array_idxs_ = self.array_map_[:, (0, 1)]  # TES array, (spat, spec)
        self.array_spat_ = self.array_map_[:, 0]
        self.array_spec_ = self.array_map_[:, 1]
        self.mce_idxs_ = self.array_map_[:, (2, 3)]  # mce_idxs, (row, col)
        self.mce_row_ = self.array_map_[:, 2]
        self.mce_col_ = self.array_map_[:, 3]

        # variables recording range of array and MCE
        if (not self.empty_flag_) and (self.len_ > 0):
            self.array_spat_llim_ = self.array_spat_.min()
            self.array_spat_ulim_ = self.array_spat_.max()
            self.array_spec_llim_ = self.array_spec_.min()
            self.array_spec_ulim_ = self.array_spec_.max()
            self.mce_row_llim_ = self.mce_row_.min()
            self.mce_row_ulim_ = self.mce_row_.max()
            self.mce_col_llim_ = self.mce_col_.min()
            self.mce_col_ulim_ = self.mce_col_.max()
        else:
            self.array_spat_llim_ = self.array_spat_ulim_ = -1
            self.array_spec_llim_ = self.array_spec_ulim_ = -1
            self.mce_row_llim_ = self.mce_row_ulim_ = -1
            self.mce_col_llim_ = self.mce_col_ulim_ = -1

        self.__check()

    def __repr__(self):
        return super(ArrayMap, self).__repr__() + ", band: " + (
            "unset" if not self.band_flag_ else "%i micron" % self.band_)

    def __check(self):
        """
        Check for length consistency and duplicate entries and mappings.

        :raises ValueError: inconsistent length
        """

        if self.empty_flag_ or (self.len_ == 0):
            self.pix_num_ = 0
            return

        if self.len_ != self.array_map_.shape[0]:
            raise ValueError("Inconsistent length found.")
        # check duplicate entry
        array_map_tuple = map(tuple, self.array_map_)
        freq_dict = Counter(array_map_tuple)
        for (item, freq) in freq_dict.items():
            if freq > 1:
                warnings.warn("Duplicate entry: %s" % str(item), UserWarning)
        # check duplicate MCE mapping
        mce_idxs_tuple = map(tuple, self.mce_idxs_)
        freq_dict = Counter(mce_idxs_tuple)
        for (item, freq) in freq_dict.items():
            if freq > 1:
                warnings.warn("Duplicate mapping from MCE: %s" % str(item),
                              UserWarning)
        # check duplicate array mapping
        pix_num = 0  # number of unique pixels mapped from MCE
        array_idxs_tuple = map(tuple, self.array_idxs_)
        freq_dict = Counter(array_idxs_tuple)
        for (item, freq) in freq_dict.items():
            pix_num += 1
            if freq > 1:
                warnings.warn("Duplicate mapping to TES array: %s" % str(item),
                              UserWarning)
        self.pix_num_ = pix_num

    def __eq__(self, other):
        """
        Compare the array_map_, band_flag_ and band_ instance with another
        ArrayMap object, return True if all instances are exactly the same
        values.

        :param ArrayMap other: ArrayMap, the ArrayMap object to compare
            with
        :return id_flag: flag whether the current array map is identical to the
            input array map
        :rtype: bool
        """

        same_flag = super(ArrayMap, self).__eq__(other)
        if same_flag:
            same_flag = same_flag & (self.band_flag_ ==
                                     other.band_flag_)
            same_flag = same_flag & (self.band_ == other.band_)
            same_flag = same_flag & np.all(self.array_map_ ==
                                           other.array_map_)

        return same_flag

    def __contains__(self, other):
        other = self.__class__(arr_in=other)
        if (not other.empty_flag_) and (other.len_ > 0):
            return np.all(np.apply_along_axis(lambda arr: np.any(
                    ~np.any((self.array_idxs_ - arr).astype(bool), axis=1)),
                                              axis=1, arr=other.array_idxs_))
        else:
            return True

    def set_band(self, band=None):
        """
        Specify the band used for the array map and truncate array map given
        the array configuration saved in conf_kwargs_ or the prior knowledge of
        TES array layout. This is necessary if you would like to initialize the
        wavelength of the array map. The function will first search in
        self.conf_kwargs_ for the input band, if spec_llim or/and spec_ulim
        or/and spat_llim or/and spat_ulim exist in the corresponding section(s)
        in self.conf_kwargs_, the array will be truncated accordingly; if
        none of the parameters exist, or the section for the input band is not
        found, the function will fall back to the
        default layout:
            =========  =====  =====  =====  =====  =====
            band       200    350    450    400    600
            ---------  -----  -----  -----  -----  -----
            spec llim  0      0      20     0      0
            spec ulim  23     19     39     39     11
            spat ulim  9      8      8      8      5
            spat llim  0      0      0      0      0
            =========  =====  =====  =====  =====  =====

        :param int or None band: int or None, if the input band can not be found
            in the keys in conf_kwargs_ attribute, the default acceptable values
            are 200, 350, 450, 400, 600; if left None, the band information will
            be reset and band_flag_ will be set to False
        :raises ValueError: invalid input of band
        """

        # check whether to reset
        if band is None:
            self.band_ = ArrayMap.band_
            self.band_flag_ = False
            print("Band reset.")
        else:
            band = int(band)
            extents = []  # list of [(spec_llim, spec_ulim, spat_ulim, spat_llim),
            # ..]

            # check in conf_kwargs_
            conf_kwargs_list, conf_flag = [], True
            for key in self.conf_kwargs_:  # looking for sections matching band
                if (str(band) in key) and (type(self.conf_kwargs_[key]) is dict):
                    conf_kwargs_use = self.conf_kwargs_.copy()
                    conf_kwargs_use.update(self.conf_kwargs_[key])
                    conf_kwargs_list.append(conf_kwargs_use)
            if len(conf_kwargs_list) > 0:
                for conf_kwargs in conf_kwargs_list:
                    key_flag, extent = False, []
                    for key in ("spec_llim", "spec_ulim",
                                "spat_ulim", "spat_llim"):
                        if key in conf_kwargs:
                            extent.append(conf_kwargs[key])
                            key_flag = True
                        else:
                            extent.append(getattr(self, "array_%s_" % key))
                    extents.append(extent)
                    conf_flag &= key_flag
            else:
                conf_flag = False

            if not conf_flag:  # Fall back to default
                if band not in (200, 350, 450, 400, 600):
                    raise ValueError("Invalid input of band, accepted " +
                                     "values are 200, 350, 450, 400, 600")

                # set default extent of the array
                extents = [{200: (0, 23, 9, 0), 350: (0, 19, 8, 0),
                            450: (20, 39, 8, 0), 400: (0, 39, 8, 0),
                            600: (0, 11, 8, 0)}[band]]

            print("Setting band using %s." % ("array configuration" if conf_flag
                                              else "default layout"))
            mask_list = []
            for extent in extents:
                mask_list.append((self.array_spec_ > extent[0] - 0.5) &
                                 (self.array_spec_ < extent[1] + 0.5) &
                                 (self.array_spat_ < extent[2] + 0.5) &
                                 (self.array_spat_ > extent[3] - 0.5))
            flag_arr = np.any(mask_list, axis=0)
            if np.any(np.sum(mask_list, axis=0) > 1):
                warnings.warn("The matching array configurations overlap.",
                              UserWarning)
            if (self.len_ > 0) and (flag_arr.sum() == 0):
                warnings.warn("The array map is incompatible with input band.",
                              UserWarning)
            self.__fill_values__(arr_in=self.data_[flag_arr])

            self.band_ = band
            self.band_flag_ = True
            if self.wl_flag_:
                self.array_wl_ = self.array_wl_[flag_arr]
                self.array_d_wl_ = self.array_d_wl_[flag_arr]

    def take_by_flag(self, flag_arr):
        """
        Create a new object from the current ArrayMap selected by the bool array
        flag_arr.

        :param flag_arr: bool array, the length should match with the array map,
            the pixels flagged as True will be used in the new ArrayMap
        :type flag_arr: list or tuple or numpy.ndarray
        :return array_map_new: ArrayMap, a new object containing the pixels
            selected by the flags
        :rtype: ArrayMap
        :raise ValueError: inconsistent length
        """

        flag_arr = np.asarray(flag_arr, dtype=bool).flatten()
        if self.len_ != len(flag_arr):
            raise ValueError("Inconsistent length of input flag_arr.")

        array_map_new = self.take_by_flag_along_axis(flag_arr, axis=0)
        array_map_new.band_flag_ = self.band_flag_
        array_map_new.band_ = self.band_
        array_map_new.conf_kwargs_.update(self.conf_kwargs_)
        if self.wl_flag_:
            array_map_new.wl_flag_ = True
            array_map_new.array_wl_ = np.extract(
                    condition=flag_arr, arr=self.array_wl_)
            array_map_new.array_d_wl_ = np.extract(
                    condition=flag_arr, arr=self.array_d_wl_)

        return array_map_new

    def expand(self, other):
        """
        Append another array map to the end of the current one, will update the
        related instances of the current array map.

        :param ArrayMap other: ArrayMap, the array map to append to
            the current ArrayMap object
        :raises TypeError: wrong type of input
        :raises ValueError: bands do not match
        """

        # check type
        if not isinstance(other, ArrayMap):
            raise TypeError("Invalid type of input, expect ArrayMap.")
        # check band
        if (not self.empty_flag_ and not other.empty_flag_) and \
                ((self.band_flag_ != other.band_flag_) or
                 (self.band_ != other.band_)):
            raise ValueError("The bands do not match.")
        else:
            if self.conf_kwargs_ == {}:  # replace empty conf_kwargs
                self.conf_kwargs_ = other.conf_kwargs_.copy()
            elif (other.conf_kwargs_ != {}) and \
                    (self.conf_kwargs_ != other.conf_kwargs_):
                self.conf_kwargs_ = {}  # drop mismatched conf_kwargs

        self.append_along_axis(other, axis=0)
        self.__fill_values__(self.data_)
        if self.wl_flag_ and other.wl_flag_:
            self.array_wl_ = np.concatenate((self.array_wl_, other.array_wl_))
            self.array_d_wl_ = np.concatenate(
                    (self.array_d_wl_, other.array_d_wl_))

    def get_flag_where(self, spec=None, spec_ran=None, spec_list=None,
                       spat=None, spat_ran=None, spat_list=None,
                       spat_spec=None, spat_spec_list=None,
                       row=None, row_ran=None, row_list=None,
                       col=None, col_ran=None, col_list=None,
                       row_col=None, row_col_list=None,
                       logic="and"):
        """
        Find and return a bool array with the pixels in the array map meeting
        the selection criteria labeled as True. The selection criteria are value
        of spec/spat/row/col, a list(or tuple) of the range of values,
        a list(or tuple) of the values; (spat, spec)/(row, col) value, or a list
        (or tuple) of (spat, spec)/(row, col). The logical condition is
        controlled by logic, which can be 'and' or 'or', meaning pixels meeting
        all or any of the input conditions are flagged.

        :param int spec: int, the specific spectral index to search
        :param spec_ran: list or tuple or array, will search for pixels with
            spec_ran.min() <= spectral index <= spec_ran.max()
        :type spec_ran: list or tuple or numpy.ndarray
        :param spec_list: list or tuple or array, will search for pixels at the
            spectral indices in spec_list
        :type spec_list: list or tuple or numpy.ndarray
        :param int spat: int, the specific spatial position to search
        :param spat_ran: list or tuple or array, will search for pixels
            with spat_ran.min() <= spatial position <= spat_ran.max()
        :type spat_ran: list or tuple or numpy.ndarray
        :param spat_list: list or tuple or array, will search for pixels at the
            spatial positions in spat_list
        :type spat_list: list or tuple or numpy.ndarray
        :param spat_spec: list or tuple or array, (spat, spec) of the pixel
        :type spat_spec: list or tuple or numpy.ndarray
        :param spat_spec_list: list or tuple or array, a list of (spat, spec) of
            the pixels
        :type spat_spec_list: list or tuple or numpy.ndarray
        :param int row: int, the specific MCE row to search
        :param row_ran: list or tuple or array, will search for pixels with
            row_ran.min() <= MCE row <= row_ran.max()
        :type row_ran: list or tuple or numpy.ndarray
        :param list row_list: list or tuple or array, will search for pixels at
            MCE row indices in row_list
        :type row_list: list or tuple or numpy.ndarray
        :param int col: int, the specific MCE column to search
        :param col_ran: list or tuple or array, will search for pixels with
            col_ran.min() <= MCE column <= col_ran.max()
        :type col_ran: list or tuple or numpy.ndarray
        :param col_list: list or tuple or array, will search for pixels at MCE
            column indices in col_list
        :type col_list: list or tuple or numpy.ndarray
        :param row_col: list or tuple or array, (row, col) of the pixel
        :type row_col: list or tuple or numpy.ndarray
        :param row_col_list: list or tuple or array, a list of (row, col) of the
            pixels
        :type row_col_list: list or tuple or numpy.ndarray
        :param logic: str or bool or int, if input 'and'(case insensitive) or
            '&' or True or 1, then only pixels meeting all the input selection
            conditions are flagged; if input 'or' or '|' or False or 0, then
            pixels meeting any of the selection conditions are flagged
        :type logic: str or bool or int
        :return flag_arr: bool array, in which the pixels in array map that
            match any/all of the criteria are flagged as True
        :rtype: numpy.ndarray
        :raises TypeError: wrong type for input logic, spec, spat, row, col,
            spat_spec, row_col
        :raises ValueError: invalid input for logic, or wrong format for
            spat_spec_list, row_col_list, or if all search condition are None
        :example:
        flag_arr = arr_in.flag_where(spec=5,spec_ran=(0,3),spec_list=(7,10),
            logic='or')
            will return an bool array in which all pixels at spectral position
            0,1,2,3,5,7,10 are flagged
        :example:
        flag_arr = arr_in.flag_where(spat=7,spec_ran=(3,10),col_ran=(15,17),
            logic='and')
            will return bool array that flags the pixels that are at spatial
            position 7, spectral index between 3 and 10, and in MCE column 15
            through 17.
        """

        # check logic type
        if isinstance(logic, (int, float, bool, np.integer, np.double)):
            logic = bool(int(logic))
            logic = "and" if logic else "or"
        if not isinstance(logic, str):
            raise TypeError("Invalid input type for logic")
        else:
            logic = logic.lower().strip()
            if logic in ["and", "&"]:
                logical = np.logical_and
            elif logic in ["or", "|"]:
                logical = np.logical_or
            else:
                raise ValueError("Invalid input value for logic")

        # create flag according to each condition
        flag_list = []
        for (var, var_ran, var_list, array_use) in (
                (spec, spec_ran, spec_list, self.array_spec_),
                (spat, spat_ran, spat_list, self.array_spat_),
                (row, row_ran, row_list, self.mce_row_),
                (col, col_ran, col_list, self.mce_col_)):
            if var is not None:
                if not isinstance(var, (int, float, np.integer, np.double)):
                    raise TypeError("Input spec,spat,row,col should be a number")
                flag_list.append((array_use == self.dtype_.type(var)).tolist())
            if var_ran is not None:
                if not isinstance(var_ran, (list, tuple, np.ndarray)):
                    raise TypeError("Input _ran should be size 2 list or array")
                llim, ulim = np.nanmin(var_ran), np.nanmax(var_ran)
                flag_list.append([(llim <= i <= ulim) for i in array_use])
            if (var_list is not None) and (len(var_list) > 0):
                if not isinstance(var_list, (list, tuple, np.ndarray)):
                    raise TypeError("Input _list should be 1-d list or array")
                var_arr = np.array(var_list, dtype=self.dtype_)
                flag_list.append([i in var_arr for i in array_use])
        if spat_spec is not None:
            try:
                spt, spc = self.dtype_.type(spat_spec[0]), \
                           self.dtype_.type(spat_spec[1])
            except TypeError:
                raise TypeError("Invalid type or shape for input spat_spec.")
            flag_list.append(((self.array_spat_ == spt) &
                              (self.array_spec_ == spc)).tolist())
        if (spat_spec_list is not None) and (len(spat_spec_list) > 0):
            spat_spec_arr = np.array(spat_spec_list, dtype=self.dtype_)
            if spat_spec_arr.shape == (2,):
                spat_spec_arr = spat_spec_arr.reshape(1, 2)
            elif (spat_spec_arr.ndim != 2) or \
                    (spat_spec_arr.shape[-1] != 2):
                raise ValueError("Invalid format for input spat_spec_arr.")
            spt, spc = spat_spec_arr[:, 0], spat_spec_arr[:, 1]
            flag_list.append([np.any((spt == i) & (spc == j))
                              for (i, j) in self.array_idxs_])
        if row_col is not None:
            try:
                ro, co = self.dtype_.type(row_col[0]), \
                         self.dtype_.type(row_col[1])
            except TypeError:
                raise TypeError("Invalid type or shape for input row_col")
            flag_list.append(((self.mce_row_ == ro) &
                              (self.mce_col_ == co)).tolist())
        if (row_col_list is not None) and (len(row_col_list) > 0):
            row_col_arr = np.array(row_col_list, dtype=self.dtype_)
            if row_col_arr.shape == (2,):
                row_col_arr = row_col_arr.reshape(1, 2)
            elif (row_col_arr.ndim != 2) or \
                    (row_col_arr.shape[-1] != 2):
                raise ValueError("Invalid format for input row_col_list")
            ro, co = row_col_arr[:, 0], row_col_arr[:, 1]
            flag_list.append([np.any((ro == i) & (co == j))
                              for (i, j) in self.mce_idxs_])
            # mce_idxs is [row, col] order

        if len(flag_list) == 0:
            flag_arr = np.full(self.len_, fill_value=False, dtype=bool)
        else:
            flag_list_arr = np.array(flag_list, dtype=bool)
            flag_arr = flag_list_arr[0]
            for arr in flag_list_arr:
                flag_arr = logical(flag_arr, arr)

        return flag_arr

    def get_index_where(self, spec=None, spec_ran=None, spec_list=None,
                        spat=None, spat_ran=None, spat_list=None,
                        spat_spec=None, spat_spec_list=None,
                        row=None, row_ran=None, row_list=None,
                        col=None, col_ran=None, col_list=None,
                        row_col=None, row_col_list=None,
                        logic="and"):
        """
        Find the indices of the pixels in array map that match the search
        condition. The search condition follows the same rule as in flag_where()
        method.

        :param int spec: int, the specific spectral index to search
        :param spec_ran: list or tuple or array, will search for pixels with
            spec_ran.min() <= spectral index <= spec_ran.max()
        :type spec_ran: list or tuple or numpy.ndarray
        :param spec_list: list or tuple or array, will search for pixels at the
            spectral indices in spec_list
        :type spec_list: list or tuple or numpy.ndarray
        :param int spat: int, the specific spatial position to search
        :param spat_ran: list or tuple or array, will search for pixels
            with spat_ran.min() <= spatial position <= spat_ran.max()
        :type spat_ran: list or tuple or numpy.ndarray
        :param spat_list: list or tuple or array, will search for pixels at the
            spatial positions in spat_list
        :type spat_list: list or tuple or numpy.ndarray
        :param spat_spec: list or tuple or array, (spat, spec) of the pixel
        :type spat_spec: list or tuple or numpy.ndarray
        :param spat_spec_list: list or tuple or array, a list of (spat, spec) of
            the pixels
        :type spat_spec_list: list or tuple or numpy.ndarray
        :param int row: int, the specific MCE row to search
        :param row_ran: list or tuple or array, will search for pixels with
            row_ran.min() <= MCE row <= row_ran.max()
        :type row_ran: list or tuple or numpy.ndarray
        :param list row_list: list or tuple or array, will search for pixels at
            MCE row indices in row_list
        :type row_list: list or tuple or numpy.ndarray
        :param int col: int, the specific MCE column to search
        :param col_ran: list or tuple or array, will search for pixels with
            col_ran.min() <= MCE column <= col_ran.max()
        :type col_ran: list or tuple or numpy.ndarray
        :param col_list: list or tuple or array, will search for pixels at MCE
            column indices in col_list
        :type col_list: list or tuple or numpy.ndarray
        :param row_col: list or tuple or array, (row, col) of the pixel
        :type row_col: list or tuple or numpy.ndarray
        :param row_col_list: list or tuple or array, a list of (row, col) of the
            pixels
        :type row_col_list: list or tuple or numpy.ndarray
        :param logic: str or bool or int, if input 'and'(case insensitive) or
            '&' or True or 1, then only pixels meeting all the input selection
            conditions are flagged; if input 'or' or '|' or False or 0, then
            pixels meeting any of the selection conditions are flagged
        :type logic: str or bool or int
        :return idx_arr: int array, containing the index of pixel in the
            array map that matches any/all of the criteria
        :rtype: numpy.ndarray
        :example:
        idx_arr = arr_in.index_where(spec=5,spec_ran=(0,3),spec_list=(7,10),
            logic='or')
            will return an array of the indices of the pixels at spectral index
            0,1,2,3,5,7,10
        :example:
        idx_arr = arr_in.index_where(spat=7,spec_ran=(3,10),col_ran=(15,17),
            logic='and')
            will return an array of the indices the pixels that are at spatial
            position 7, spectral index between 3 and 10, and in MCE column 15
            through 17.
        """

        flag_arr = self.get_flag_where(spec=spec, spec_ran=spec_ran,
                                       spec_list=spec_list, spat=spat,
                                       spat_ran=spat_ran, spat_list=spat_list,
                                       spat_spec=spat_spec,
                                       spat_spec_list=spat_spec_list, row=row,
                                       row_ran=row_ran, row_list=row_list,
                                       col=col, col_ran=col_ran,
                                       col_list=col_list,
                                       row_col=row_col,
                                       row_col_list=row_col_list, logic=logic)
        idx_arr = np.flatnonzero(flag_arr)

        return idx_arr

    def take_where(self, spec=None, spec_ran=None, spec_list=None,
                   spat=None, spat_ran=None, spat_list=None,
                   spat_spec=None, spat_spec_list=None,
                   row=None, row_ran=None, row_list=None,
                   col=None, col_ran=None, col_list=None,
                   row_col=None, row_col_list=None,
                   logic="and"):
        """
        Get a new object of ArrayMap according to the selection criteria.
        The search condition follows the same rule as in flag_where()
        method.

        :param int spec: int, the specific spectral index to search
        :param spec_ran: list or tuple or array, will search for pixels with
            spec_ran.min() <= spectral index <= spec_ran.max()
        :type spec_ran: list or tuple or numpy.ndarray
        :param spec_list: list or tuple or array, will search for pixels at the
            spectral indices in spec_list
        :type spec_list: list or tuple or numpy.ndarray
        :param int spat: int, the specific spatial position to search
        :param spat_ran: list or tuple or array, will search for pixels
            with spat_ran.min() <= spatial position <= spat_ran.max()
        :type spat_ran: list or tuple or numpy.ndarray
        :param spat_list: list or tuple or array, will search for pixels at the
            spatial positions in spat_list
        :type spat_list: list or tuple or numpy.ndarray
        :param spat_spec: list or tuple or array, (spat, spec) of the pixel
        :type spat_spec: list or tuple or numpy.ndarray
        :param spat_spec_list: list or tuple or array, a list of (spat, spec) of
            the pixels
        :type spat_spec_list: list or tuple or numpy.ndarray
        :param int row: int, the specific MCE row to search
        :param row_ran: list or tuple or array, will search for pixels with
            row_ran.min() <= MCE row <= row_ran.max()
        :type row_ran: list or tuple or numpy.ndarray
        :param list row_list: list or tuple or array, will search for pixels at
            MCE row indices in row_list
        :type row_list: list or tuple or numpy.ndarray
        :param int col: int, the specific MCE column to search
        :param col_ran: list or tuple or array, will search for pixels with
            col_ran.min() <= MCE column <= col_ran.max()
        :type col_ran: list or tuple or numpy.ndarray
        :param col_list: list or tuple or array, will search for pixels at MCE
            column indices in col_list
        :type col_list: list or tuple or numpy.ndarray
        :param row_col: list or tuple or array, (row, col) of the pixel
        :type row_col: list or tuple or numpy.ndarray
        :param row_col_list: list or tuple or array, a list of (row, col) of the
            pixels
        :type row_col_list: list or tuple or numpy.ndarray
        :param logic: str or bool or int, if input 'and'(case insensitive) or
            '&' or True or 1, then only pixels meeting all the input selection
            conditions are flagged; if input 'or' or '|' or False or 0, then
            pixels meeting any of the selection conditions are flagged
        :type logic: str or bool or int
        :return: ArrayMap object, in which only the pixels in the
            current array map that match any/all of the criteria are present. The
            band and origin will be passed to the new ArrayMap object
        :rtype: ArrayMap
        :example:
        array_map_new = arr_in.cut_where(spec=5,spec_ran=(0,3),
            spec_list=(7,10),logic='or')
            will return a new ArrayMap object containing all the pixels at
            spectral position 0,1,2,3,5,7,10 in the original array map.
        :example:
        array_map_new = arr_in.flag_where(spat=7,spec_ran=(3,10),
            col_ran=(15,17),logic='and')
            will return a new ArrayMap object containing the pixels that are at
            spatial position 7, spectral index between 3 and 10, and in MCE
            column 15 through 17 in the original array map.
        """

        flag_arr = self.get_flag_where(spec=spec, spec_ran=spec_ran,
                                       spec_list=spec_list, spat=spat,
                                       spat_ran=spat_ran, spat_list=spat_list,
                                       spat_spec=spat_spec,
                                       spat_spec_list=spat_spec_list, row=row,
                                       row_ran=row_ran, row_list=row_list,
                                       col=col, col_ran=col_ran,
                                       col_list=col_list,
                                       row_col=row_col,
                                       row_col_list=row_col_list,
                                       logic=logic)

        return self.take_by_flag(flag_arr=flag_arr)

    def exclude_where(self, spec=None, spec_ran=None, spec_list=None,
                      spat=None, spat_ran=None, spat_list=None,
                      spat_spec=None, spat_spec_list=None,
                      row=None, row_ran=None, row_list=None,
                      col=None, col_ran=None, col_list=None,
                      row_col=None, row_col_list=None,
                      logic="and"):
        """
        Get a new object of ArrayMap that excludes the pixels according to
        the selection criteria. The keywords are the same as
        ArrayMap.take_where().

        :return: ArrayMap object, in which only the pixels in the
            current array map that match any or all of the criteria are present.
            The band and origin will be passed to the new ArrayMap object
        :rtype: ArrayMap
        """

        flag_arr = self.get_flag_where(spec=spec, spec_ran=spec_ran,
                                       spec_list=spec_list, spat=spat,
                                       spat_ran=spat_ran, spat_list=spat_list,
                                       spat_spec=spat_spec,
                                       spat_spec_list=spat_spec_list, row=row,
                                       row_ran=row_ran, row_list=row_list,
                                       col=col, col_ran=col_ran,
                                       col_list=col_list,
                                       row_col=row_col,
                                       row_col_list=row_col_list,
                                       logic=logic)

        return self.take_by_flag(flag_arr=~flag_arr)

    def index_sort(self, keys):
        """
        Return the indices of the pixels in array map sorted according to one or
        more keys, in the order of keys input.

        :param keys: str or int or list or tuple or array of str or int,
            allowed str input are 'spat', 'spec', 'row', 'col' (case insensitive),
            int input can be 0, 1, 2, 3 representing the second axis of
            array_map_ instance. The data will be sorted in the order of keys.
            For example, if keys=('spat', 'spec'), all the returned indices will
            be sorted by 'spat', and at any given value of 'spat', the indices
            are sorted by 'spec'
        :type keys: str or int or list or tuple or numpy.ndarray
        :return idx_arr: array, containing the indices of pixels sorted according
            to keys
        :rtype: numpy.ndarray
        :example:
        idx_arr = ArrayMap(arr_in=[[0,0,0,0], [0,1,1,0], [0,2,0,1],
        [1,0,1,1]]).index_sort(keys=('row', 'spec', 'spat'))
        will return [0, 2, 3, 1]
        """

        idx_arr = self.pix_idxs_.copy()
        axis_dict = {'spat': 0, 'spec': 1, 'row': 2, 'col': 3}
        if isinstance(keys, (str, int, float, np.integer, np.double)):
            keys = [keys]
        for key in reversed(keys):  # sort backwards in keys
            if isinstance(key, (int, float, np.integer, np.double)):
                ax = int(key)
            elif isinstance(key, str):
                try:
                    ax = axis_dict[key.lower().strip()]
                except KeyError:
                    raise ValueError('Invalid input: %s' % key)
            else:
                raise ValueError('Invalid input type: %s' % str(type(key)))
            if ax not in (0, 1, 2, 3):
                raise ValueError('Invalid input: %s' % ax)
            arg_sort = self.array_map_[idx_arr, ax].argsort(kind='mergesort')
            idx_arr = idx_arr[arg_sort]

        return idx_arr

    def sort(self, keys):
        """
        return the array map according to one or more keys, in the order of keys
        input. Follows the same sorting order as in index_sort method().

        :param keys: str or int or list or tuple or array of str or int,
            allowed str input are 'spat', 'spec', 'row', 'col' (case insensitive),
            int input can be 0, 1, 2, 3 representing the second axis of
            array_map_ instance. The data will be sorted in the order of keys.
            For example, if keys=('spat', 'spec'), all the returned indices will
            be sorted by 'spat', and at any given value of 'spat', the indices
            are sorted by 'spec'
        :type keys: str or int or list or tuple or numpy.ndarray
        :return arr_map_new: ArrayMap object that is sorted accordingly
        :rtype: ArrayMap
        :example:
        arr_map_new = ArrayMap(arr_in=[[0,0,0,0], [0,1,1,0], [0,2,0,1],
        [1,0,1,1]]).sort(keys=(2, 'spec', 'spat'))
        will get an array map of [[0,0,0,0], [0,2,0,1], [1,0,1,1], [0,1,1,0]]
        """

        idx_arr = self.index_sort(keys=keys)
        array_map_new = self.replace(arr_in=self.data_[idx_arr])
        array_map_new.band_flag_ = self.band_flag_
        array_map_new.band_ = self.band_
        array_map_new.conf_kwargs_.update(self.conf_kwargs_)
        if self.wl_flag_:
            array_map_new.wl_flag_ = True
            array_map_new.array_wl_ = self.array_wl_[idx_arr]
            array_map_new.array_d_wl_ = self.array_d_wl_[idx_arr]

        return array_map_new

    def get_conf(self, **kwargs):
        """
        Assemble a list of dictionaries with parameters in conf_kwargs_ matching
        the band_ of the object; the input keyword arguments will override any
        value saved in conf_kwargs_

        :return: [conf_kwargs1, [conf_kwargs2], ...] list of dict of keyword
            arguments; if the array map band corresponds none or only one
            section in conf_kwargs_, the return list will only contain one item,
            otherwise more than one kwargs items will be in the returned list
            which is the case of 400 um array
        :rtype: list[dict]
        """

        conf_kwargs, conf_kwargs_list = self.conf_kwargs_.copy(), []

        if not self.band_flag_:  # update kwargs according to band
            warnings.warn("ArrayMap band not set, will use the input " +
                          "arguments and all available bands in conf_kwargs_.",
                          UserWarning)
            band = str()
        else:
            band = str(self.band_)

        for key in conf_kwargs:  # looking for sections matching band
            if (band in key) and (type(conf_kwargs[key]) is dict):
                conf_kwargs_use = conf_kwargs.copy()
                conf_kwargs_use.update(conf_kwargs[key])
                conf_kwargs_use.update(kwargs)
                conf_kwargs_list.append(conf_kwargs_use)

        if len(conf_kwargs_list) == 0:  # build one conf_kwargs
            warnings.warn("No band specific config information found, will use " +
                          "the input kwargs only the DEFAULT configurations " +
                          "in conf_kwargs_.", UserWarning)
            conf_kwargs_use = conf_kwargs.copy()
            conf_kwargs_use.update(kwargs)
            conf_kwargs_list.append(conf_kwargs_use)
        elif len(conf_kwargs_list) > 1:  # remove redundant
            pop_idx = []
            for i, conf_kwargs1 in enumerate(conf_kwargs_list[:-1]):
                for j, conf_kwargs2 in enumerate(conf_kwargs_list[i + 1:]):
                    if conf_kwargs1 == conf_kwargs2:
                        pop_idx.append(j + i + 1)
            for idx in np.sort(pop_idx)[::-1]:
                conf_kwargs_list.pop(idx)

        return conf_kwargs_list

    def init_wl(self, **kwargs):
        """
        Converting the spectral indices to wavelength in micron using
        tools.spec_to_wl() function. The function will calculate the wavelength
        and wavelength interval of each pixel, recorded in self.array_wl_ and
        self.array_d_wl_ variables, and changing self.wl_flag_ to True. The
        kwargs passed to tools.spec_to_wl() will combine the parameters in the
        input kwargs and self.conf_kwargs by calling self.get_conf(), following
        the priority input keyword argument > self.conf_kwargs.

        :param dict kwargs: keyword arguments passed to tools.spec_to_wl()
            except spec and spat, for a list of accepted parameters, please
            refer to the function tools.spec_to_wl(); remember to supply
            grat_idx parameter if it is not set in self.conf_kwargs
        """

        conf_kwargs_list = self.get_conf(**kwargs)
        if len(conf_kwargs_list[0]) == 0:
            warnings.warn("empty configuration, will use the default values " +
                          "spec_to_wl() function.", UserWarning)
        wl, d_wl = np.full(self.len_, dtype=float, fill_value=np.nan), \
                   np.full(self.len_, dtype=float, fill_value=np.nan)
        func_vars = inspect.getfullargspec(spec_to_wl)[0]

        for conf_kwargs in conf_kwargs_list:
            kwargs_use = conf_kwargs.copy()
            for key in conf_kwargs:
                if (key in ("spec", "spat")) or (key not in func_vars):
                    kwargs_use.pop(key)
            wl_use = spec_to_wl(spec=self.array_spec_, spat=self.array_spat_,
                                **kwargs_use)
            d_wl_use = \
                spec_to_wl(spec=self.array_spec_ + 0.5, spat=self.array_spat_,
                           **kwargs_use) - \
                spec_to_wl(spec=self.array_spec_ - 0.5, spat=self.array_spat_,
                           **kwargs_use)

            extent = []
            for key in ("spec_llim", "spec_ulim", "spat_ulim", "spat_llim"):
                if key in conf_kwargs:
                    extent.append(conf_kwargs[key])
                else:
                    extent.append(getattr(self, "array_%s_" % key))
            wl_mask = (self.array_spec_ > extent[0] - 0.5) & \
                      (self.array_spec_ < extent[1] + 0.5) & \
                      (self.array_spat_ < extent[2] + 0.5) & \
                      (self.array_spat_ > extent[3] - 0.5)
            np.putmask(wl, wl_mask, wl_use)
            np.putmask(d_wl, wl_mask, d_wl_use)

        self.wl_flag_ = True
        self.array_wl_ = wl
        self.array_d_wl_ = abs(d_wl)

    def read_conf(self, filename):
        """
        Initialize conf_kwargs_ variable by reading in the configuration
        parameters .ini file. The format of the configuration file should follow
        Windows Registry extended version of INI syntax, and the section keyword
        should be band wavelength, such as 200, 350, 400 etc. The current
        conf_kwargs_ variable will be updated with the new variables and values

        :param str filename: str, path to the file containing
        """

        if not os.path.isfile(filename):
            warnings.warn("input %s does not exist." % filename, UserWarning)
        else:
            config = configparser.ConfigParser()
            config.read(filename)

            conf_kwargs = self.conf_kwargs_.copy()
            for section in config:
                if section != "DEFAULT":
                    conf_kwargs[section] = {}
                for key in config[section]:
                    try:
                        value = float(config[section][key])
                    except ValueError:
                        value = config[section][key]
                    if section == "DEFAULT":
                        conf_kwargs[key] = value
                    else:
                        conf_kwargs[section][key] = value

            self.conf_kwargs_ = conf_kwargs


class Chop(DataObj):
    """
    chop phase of the data. Can be initialized by a 1-d bool or int array or
        list, or use read() class method to read from .chop file.
    """

    obj_type_ = "Chop"  # type str
    dtype_ = np.dtype(bool)  # type: numpy.dtype
    chunk_num_ = 0  # type: int
    chunk_edge_idxs_ = np.array([0], dtype=int)  # type: numpy.ndarray

    # output of index_diff_edge(), recording edge indices of each chunk

    def __init__(self, arr_in=None, chunk_edge_idxs=None):
        """
        Initialize chop by numpy array or list.

        :param arr_in: list or tuple or array or Chop object, 1 dimension of
            bool type
        :type arr_in: list or tuple or numpy.ndarray or Chop
        :raises TypeError: wrong input type for arr_in
        :raises ValueError: no input, wrong dimension for input arr_in
        :example:
        chop = Chop(arr_in=[1, 1, 0, 0, 1])
        """

        super(Chop, self).__init__(arr_in=arr_in)
        if isinstance(arr_in, type(self)) and isinstance(self, type(arr_in)):
            pass
        elif isinstance(arr_in, Chop):
            self.update_chunk(chunk_edge_idxs=arr_in.chunk_edge_idxs_)
        else:
            if self.ndim_ != 1:
                raise ValueError('Invalid dimension for Chop input arr_in.')
            if chunk_edge_idxs is None:
                chunk_edge_idxs = self.get_index_chunk_edge(thre_min=0.5,
                                                            thre_max=None)
            self.update_chunk(chunk_edge_idxs=chunk_edge_idxs)

    def replace(self, **kwargs):
        """
        Initialize a new Chop with the input arr_in but copy all other
        instance variable values. A wrapper method for __operate

        :rtype: Chop
        """

        for kw in ["chunk_edge_idxs"]:
            if kw not in kwargs:
                kwargs[kw] = None

        return super(Chop, self).replace(**kwargs)

    @classmethod
    def read(cls, filename):
        """
        Return a chop object initialized by a .chop file

        :param str filename: str, path to the .chop file containing chop, will try
            to read filename.chop if the first read attempt fails
        :return chop: Chop, new object
        :rtype: Chop

        :example:
        chop = Chop.read(filename='irc10216_191128_0065.chop')
        """

        if filename is None:
            arr_in = None
        else:
            try:
                arr_in = Tb.read(filename, format="ascii")["col2"]. \
                    data.astype(int)
            except FileNotFoundError:
                try:
                    arr_in = Tb.read(
                            filename + ".chop", format="ascii")["col2"]. \
                        data.astype(int)
                except FileNotFoundError:
                    raise FileNotFoundError("%s or %s.chop does not exist" %
                                            (filename, filename))
        chop = cls(arr_in=arr_in)

        return chop

    def __eq__(self, other):
        """
        Compare the DataObj instance variables as well as chunk_edge_idxs_ with
        another Chop object, return True if all instances are exactly the same
        values

        :param Chop other: Chop, the chop object to compare with
        :return id_flag: flag whether the current chop is identical to the input
            chop
        :rtype: bool
        """

        same_flag = super(Chop, self).__eq__(other=other)
        if same_flag:
            same_flag = same_flag & np.all(self.chunk_edge_idxs_ ==
                                           other.chunk_edge_idxs_)

        return same_flag

    def update_chunk(self, chunk_edge_idxs):
        """
        Update chunk_edge_idxs_ and chunk_num_ instance by the input

        :param numpy.ndarray chunk_edge_idxs: array, recording edge indices of
            each chunk, output of index_diff_edge()
        :raises ValueError: invalid chunk_edge_idxs input
        """

        if (chunk_edge_idxs[0] != 0) or (chunk_edge_idxs[-1] != self.len_):
            raise ValueError("Invalid input chunk_edge_idxs format.")
        self.chunk_edge_idxs_ = chunk_edge_idxs
        self.chunk_num_ = len(chunk_edge_idxs) - 1

    def take_by_flag(self, flag_arr):
        """
        Create a new object from the current Chop selected by the bool array
        flag

        :param flag_arr: bool array, the length should match with the chop, the
            data flagged as True will be used in the new Chop
        :type flag_arr: list or tuple or numpy.ndarray
        :return chop_new: Chop, a new object containing the data flagged
        :rtype: Chop
        :raise ValueError: inconsistent length
        """

        # cumsum then take edge values
        chunk_edge_idxs_new = [0] + flag_arr. \
            cumsum(dtype=int)[self.chunk_edge_idxs_[1:] - 1].tolist()
        chunk_edge_idxs_new = np.unique(chunk_edge_idxs_new)
        chunk_edge_idxs_new.sort()
        chop_new = super(Chop, self).take_by_flag_along_axis(
                flag_arr=flag_arr, axis=0, chunk_edge_idxs=chunk_edge_idxs_new)

        return chop_new

    def take_by_flag_along_axis(self, flag_arr, axis=-1, **kwargs):
        return self.take_by_flag(flag_arr=flag_arr)

    def append(self, other):
        """
        Append another chop into the current chop, and update the
        chunk_edge_idxs_.

        :param Chop other: Chop, the chop to append to the current Chop
            object
        :raises TypeError: wrong type of input
        """

        # check type
        if not isinstance(other, Chop):  # check type
            raise TypeError("Invalid type of input, expect Chop")

        chunk_edge_idxs_new = np.concatenate((self.chunk_edge_idxs_, self.len_ +
                                              other.chunk_edge_idxs_[1:]))
        self.append_along_axis(other=other, axis=0)
        self.update_chunk(chunk_edge_idxs=chunk_edge_idxs_new)

    def get_flag_chunk_edges(self, ncut=0.05):
        """
        Return a bool array in which edge of both the beginning and the end of
        each chop phase(chunk) are flagged as True. ncut can be ratio or number
        of data points.

        :param ncut: int or float, ratio(float, <= 0.5) of chop phase(chunk) or
            number of data points(int, non-negative) to flag
        :type ncut: int or float
        :return chunk_edges_flag: array, bool array where edges of each chop
            phase are flagged
        :rtype: numpy.ndarray
        :raises ValueError: invalid value of edge
        :raises TypeError: invalid type of edge
        :example:
        flag_arr = Chop(chop_arr=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]).\
        flag_chunk_edges(ncut=0.2)
            will return array([True, False, False, False, True, True, False,
            True, False, False])
        flag_arr = Chop(chop_arr=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]).\
        flag_chunk_edges(ncut=2)
            will return array([True, True, False, True, True, True, True, True,
            True, True])
        """

        # check input
        if isinstance(ncut, (int, np.integer)):
            if ncut < 0:
                raise ValueError('Invalid value of ncut: %i' % ncut)
        elif isinstance(ncut, (float, np.double)):
            if not (0. <= ncut <= 0.5):
                raise ValueError('Invalid value of ncut: %f' % ncut)
        else:
            raise TypeError('Invalid type of ncut: %s' % str(type(ncut)))

        chunk_edges_flag = np.zeros(self.len_, dtype=bool)  # empty flag
        for i, (idx_i, idx_e) in \
                enumerate(zip(self.chunk_edge_idxs_[:-1],
                              self.chunk_edge_idxs_[1:])):
            chunk_len = idx_e - idx_i
            if isinstance(ncut, (float, np.double)):
                n = int(round(ncut * chunk_len))
            else:
                n = min(chunk_len, ncut)
            chunk_edges_flag[idx_i:idx_i + n] = True
            chunk_edges_flag[idx_e - n:idx_e] = True

        return chunk_edges_flag

    def get_flag_edge_chunks(self, ncut=3):
        """
        Return a bool array in which data in the chop phases(chunks) at both the
        beginning and the end of the data are flagged as True. Be cautious that
        this method is different from the previous version, as here ncut is the
        number of chunks, which used to be number of pairs of chunks.

        :param int ncut: int, number of chop phases(chunks) at both edges of the
            data to flag
        :return edge_chunk_flag: array, bool array where ncut of chop phases
            are flagged
        :rtype: numpy.ndarray
        :raises TypeError: invalid type of ncut
        :raises ValueError: invalid value of ncut
        :example:
        flag_arr = Chop(chop_arr=[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]).\
        flag_edge_chunks(ncut=2)
            will return array([ True, True, True, True, True, False, False,
            False, False, False, True, True, True])
        """

        if not isinstance(ncut, (int, float, np.integer, np.double)):
            raise TypeError('Invalid type of ncut: %s' % str(type(ncut)))
        ncut = int(ncut)
        if ncut < 0:
            raise ValueError('Invalid value of ncut: %i' % ncut)

        edge_chunks_flag = np.zeros(self.len_, dtype=bool)  # empty bool array
        ncut = min(ncut, self.chunk_num_)
        edge_chunks_flag[:self.chunk_edge_idxs_[ncut]] = True
        edge_chunks_flag[self.chunk_edge_idxs_[self.chunk_num_ - ncut]:] = True

        return edge_chunks_flag

    def chunk_split(self, chunk_edge_idxs):
        """
        Update chunk_edge_idxs_ besides the basic DataObj.chunk_split().

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func
        :return chunk_list: list, containing DataObj objects of data chunks
            split by chunk_edge_idxs
        :rtype: list
        """

        chunk_list = super(Chop, self).chunk_split(
                chunk_edge_idxs=chunk_edge_idxs)
        for i, (idx_i, idx_e, chop_chunk) in \
                enumerate(zip(chunk_edge_idxs[:-1], chunk_edge_idxs[1:],
                              chunk_list)):
            chunk_len = idx_e - idx_i  # length of each chunk
            edge_idxs_use_idxs = np.flatnonzero((chunk_edge_idxs >= idx_i) &
                                                (chunk_edge_idxs < idx_i))
            edge_idxs_use = self.chunk_edge_idxs_[edge_idxs_use_idxs] - idx_i
            chunk_edge_idxs_new = [0] + list(edge_idxs_use) + [chunk_len]
            chunk_edge_idxs_new = np.unique(chunk_edge_idxs_new)
            chunk_edge_idxs_new.sort()
            chop_chunk.update_chunk(chunk_edge_idxs=chunk_edge_idxs_new)

        return chunk_list

    def chunk_proc(self, chunk_edge_idxs, method="nanmean", keep_shape=False,
                   **kwargs):
        """
        Inherit from DataObj.chunk_proc(), also process chunk_idxs_.

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func
        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'mean',
            'median', 'sum', 'std', 'min', 'max', 'num', 'num_not_is_nan',
            'num_is_finite', the method to calculate data value in each chunk
        :param keep_shape: bool, flag whether the output should have the shape
            as the input, so that each chunk are not compressed
        :return chop_chunked: DataObj, dtype_ will be np.double for calculations
            or int for counting numbers, bool if nanmean, mean, nanmedian or
            median
        :rtype: DataObj
        :raises ValueError: invalid method value
        """

        # check method value
        chop_chunked = super(Chop, self).chunk_proc(
                chunk_edge_idxs=chunk_edge_idxs, method=method,
                keep_shape=keep_shape, **kwargs)
        if keep_shape:
            chunk_edge_idxs_new = chunk_edge_idxs
        else:
            chunk_edge_idxs_new = np.arange(chop_chunked.len_ + 1, dtype=int)
        chop_chunked.update_chunk(chunk_edge_idxs_new)
        if method in ["nanmean", "nanmedian", "mean", "median"]:
            chop_chunked.update_type(bool)

        return chop_chunked


class TimeStamps(DataObj):
    """
    Object that contains data time stamps read from .ts file. Can be initialized
         by a 1d float array or list, or use read() class method to read from
         .ts file.
    """

    obj_type_ = "TimeStamps"  # type str
    dtype_ = np.dtype(np.double)  # type: numpy.dtype
    interv_ = np.double(0.)  # type: numpy.double # minimal time interval
    t_start_ = np.double(-1.)  # type: numpy.double # start time of time stamp
    t_end_ = np.double(-1.)  # type: numpy.double # end time of time stamp
    t_mid_ = np.double(-1.)  # type: numpy.double # middle of the time
    t_start_time_ = Time(-1., format="unix")  # type: Time
    t_end_time_ = Time(-1., format="unix")  # type: Time
    t_mid_time_ = Time(-1., format="unix")  # type: Time

    @classmethod
    def read(cls, filename):
        """
        Return a TimeStamps object by reading from a .ts file.

        :param str filename: str, path to the .ts file containing time stamps,
            will try to read filename.ts if the first attempt fails
        :return ts: TimeStamps, new object
        :rtype: TimeStamps
        :raises RuntimeError: failed to read filename and filename.ts
        :raises FileNotFoundError: filename and filename.ts don't exist

        :example:
        ts = TimeStamps.read(filename='irc10216_191128_0065.ts')
        """
        if filename is None:
            arr_in = None
        else:
            if os.path.isfile(filename):
                try:
                    arr_in = Tb.read(filename, format="ascii.no_header",
                                     fast_reader=True, delimiter=" ",
                                     guess=False)["col2"]. \
                        data.astype(np.double)
                except Exception as err:
                    if os.path.isfile(filename + ".ts"):
                        try:
                            arr_in = Tb.read(filename + ".ts",
                                             format="ascii.no_header",
                                             fast_reader=True, delimiter=" ",
                                             guess=False)["col2"]. \
                                data.astype(np.double)
                        except Exception as err1:
                            raise RuntimeError(("failed to read %s and %s.ts." %
                                                (filename, filename))) from err1
                    else:
                        raise RuntimeError(
                                "failed to read %s." % filename) from err
            elif os.path.isfile(filename + ".ts"):
                try:
                    arr_in = Tb.read(filename + ".ts",
                                     format="ascii.no_header",
                                     fast_reader=True, delimiter=" ",
                                     guess=False)["col2"]. \
                        data.astype(np.double)
                except Exception as err2:
                    raise RuntimeError(
                            "failed to read %s.ts, and %s doesn't exist." %
                            (filename, filename)) from err2
            else:
                raise FileNotFoundError("%s and %s.ts don't exist." %
                                        (filename, filename))
        ts = cls(arr_in=arr_in)
        ts.check()

        return ts

    def __fill_values__(self, arr_in):
        """
        Assign values to the instance variables.

        :param numpy.ndarray arr_in: array, containing time stamps
        """

        super(TimeStamps, self).__fill_values__(arr_in=arr_in)

        if self.ndim_ != 1:
            raise ValueError('Invalid dimension for input arr_in.')
        if self.empty_flag_ or self.len_ < 2:
            self.interv_ = np.double(0.)
        else:
            self.interv_ = np.nanmin(abs(np.diff(self.data_)))
        if self.empty_flag_:
            self.t_start_ = self.t_end_ = np.double(-1.)
        else:
            self.t_start_ = np.nanmin(self.data_)
            self.t_end_ = np.nanmax(self.data_)
        self.t_mid_ = (self.t_start_ + self.t_end_) / 2
        self.t_start_time_ = Time(self.t_start_, format="unix")
        self.t_end_time_ = Time(self.t_end_, format="unix")
        self.t_mid_time_ = Time(self.t_mid_, format="unix")

    def __repr__(self):
        return super(TimeStamps, self).__repr__() + \
               "\n\t start: %s  end: %s" % \
               (self.t_start_time_.to_value(format="iso"),
                self.t_end_time_.to_value(format="iso"))

    def check(self):
        """
        Check if there is any problem in TimeStamp, such as t=0 and extra time
        stamps than data and chop.

        :param
        """

        if (not self.empty_flag_) and (self.len_ > 0):
            if np.any(~np.isfinite(self.data_)):
                raise ValueError('Invalid value in TimeStamp.')
            if (self.t_start_ < 1e-5) or np.any(self.data_ < 1e-5):
                warnings.warn("Time = 0 exists in TimeStamp.")
            if (self.data_.min() != self.t_start_) or \
                    (self.data_.max() != self.t_end_):
                warnings.warn("Probably wrong order of TimeStamp.")

    def append(self, other):
        """
        Append another TimeStamp after the current one, update the data_ of the
        current object

        :param TimeStamp other: TimeStamp, object to be appended
        """
        return self.append_along_axis(other=other, axis=0)

    def corr_by_chop(self, chop):
        """
        Correct time stamps by chop. Time stamp file is known to display two
        problems: extra time stamps and occasional 0 values, will call
        rebuild_by_chop() method if autocorrection fails

        :param Chop chop: Chop, chop object used as reference for correction
        :return: TimeStamps, corrected TimeStamps, will return self if no
            problem is found
        :rtype: TimeStamps
        """

        chop = Chop(arr_in=chop)
        if chop.len_ > self.len_:
            warnings.warn("Chop length greater than ts.", UserWarning)
            return self.rebuild_by_chop(chop)
        chop_chunk_edge_idxs = chop.chunk_edge_idxs_
        ts_chunk_edge_idxs = self.get_index_chunk_edge(
                thre_min=self.interv_ * 2, thre_max=self.t_mid_ / 2)
        flag_zero = self.data_ < 1e-5
        if (np.count_nonzero(flag_zero) == 0) and \
                np.all(ts_chunk_edge_idxs == chop_chunk_edge_idxs):
            return self
        else:
            # if there are extra time stamps but no 0 values
            if (np.count_nonzero(flag_zero) == 0) and \
                    np.any(ts_chunk_edge_idxs != chop_chunk_edge_idxs):
                if chop.len_ == self.len_:
                    return self  # do nothing if lengths agree
                else:
                    if len(ts_chunk_edge_idxs) > 2:  # if chop_effcy != 1
                        if len(ts_chunk_edge_idxs) - 1 != chop.chunk_num_:
                            warnings.warn("Chunk number disagree, " +
                                          "needs human inspection", UserWarning)
                            return self.rebuild_by_chop(chop)
                        ts_new = np.empty(chop.len_, dtype=self.dtype_)
                        for (idx_i_chop, idx_e_chop, idx_i, idx_e) in zip(
                                chop_chunk_edge_idxs[:-1],
                                chop_chunk_edge_idxs[1:],
                                ts_chunk_edge_idxs[:-1],
                                ts_chunk_edge_idxs[1:]):
                            if idx_e_chop - idx_i_chop > idx_e - idx_i:
                                warnings.warn(
                                        "Ts Chunk length smaller than " +
                                        "chop, needs human inspection",
                                        UserWarning)
                                return self.rebuild_by_chop(chop)
                            ts_new[idx_i_chop:idx_e_chop] = \
                                self.data_[
                                idx_i:idx_i + idx_e_chop - idx_i_chop]
                        warnings.warn("Extra time stamps truncated")
                        return TimeStamps(arr_in=ts_new)
                    else:  # if chop_effcy == 1
                        return TimeStamps(arr_in=self.data_[:chop.len_])
            # if there are 0 value elements but no extra time stamps
            elif (np.count_nonzero(flag_zero) > 0) and (chop.len_ == self.len_):
                ts_new = self.data_.copy()
                if self.interv_ != 0:
                    for idx in np.flatnonzero(flag_zero):
                        if idx not in chop_chunk_edge_idxs:
                            ts_new[idx] = ts_new[idx - 1] + self.interv_
                        else:
                            i = 1
                            while (ts_new[idx + i] < 1e-5) and \
                                    ((idx + i + 1) < len(ts_new)):
                                i += 1
                            if (idx + i + 1 == len(ts_new) - 1) and \
                                    (ts_new[idx + i] < 1e-5):
                                if idx - 1 >= 0:
                                    ts_new[idx] = ts_new[idx - 1] + self.interv_
                            else:
                                ts_new[idx] = ts_new[idx + i] - self.interv_ * i
                    warnings.warn("Zero value time stamps corrected")
                return TimeStamps(arr_in=ts_new)
            else:
                warnings.warn("0 Values and extra length in ts, " +
                              "needs human inspection", UserWarning)
                return self.rebuild_by_chop(chop)

    def rebuild_by_chop(self, chop):
        """
        rebuild the time stamp in the case that time stamp has complicated
        problems and can not be corrected automatically

        :param Chop chop: Chop, chop object used as reference for correction
        :return: TimeStamps, rebuilt TimeStamps
        :rtype: TimeStamps
        """

        ts_flag = (~np.isfinite(self.data_)) | (self.data_ == 0)
        interv_diff = abs(np.diff(self.data_[~ts_flag]))
        if len(interv_diff) <= 2:  # find the correct interval and chunk interval
            interv, chunk_interv = 0, 0
        else:
            interv = np.nanmin(interv_diff)
            if np.count_nonzero(interv_diff > interv * 2) > 0:
                chunk_interv = np.nanmedian(
                        interv_diff[interv_diff > interv * 2])
            else:
                chunk_interv = interv
        chunk_interv -= interv

        # find which part of ts is still usable
        chop_chunk_edge_idxs = chop.chunk_edge_idxs_
        ts_chunk_edge_idxs = self.get_index_chunk_edge(
                thre_min=interv * 2, thre_max=self.t_mid_ / 2)
        chunk_edge_num = min(len(chop_chunk_edge_idxs), len(ts_chunk_edge_idxs))
        if (len(np.nonzero(chop_chunk_edge_idxs[:chunk_edge_num] -
                           ts_chunk_edge_idxs[:chunk_edge_num])[0]) == 0) and \
                (chunk_edge_num == len(chop_chunk_edge_idxs)):  # only truncate ts
            ts_new = self.data_[:chop.len_]
        else:  # rebuild
            if len(np.nonzero(chop_chunk_edge_idxs[:chunk_edge_num] -
                              ts_chunk_edge_idxs[:chunk_edge_num])[0]) == 0:
                # time stamp shorter than chop
                idx_chunk_i = chunk_edge_num - 1
            else:  # rebuild time stamp from the middle
                idx_chunk_i = \
                    np.nonzero(chop_chunk_edge_idxs[:chunk_edge_num] -
                               ts_chunk_edge_idxs[:chunk_edge_num])[0][0] - 1
            idx_i = chop_chunk_edge_idxs[idx_chunk_i]
            if idx_i == 0:  # estimate the starting time
                if ts_flag.sum() == self.len_:
                    t_start = 0
                else:
                    t_finite_i = np.nonzero(~ts_flag)[0][0]
                    t_start = self.data_[t_finite_i] - interv * t_finite_i - \
                              ((chop_chunk_edge_idxs <= t_finite_i).sum() - 1) * \
                              chunk_interv
            else:
                t_start = self.data_[idx_i - 1]
            ts_new = np.empty(chop.shape_, dtype=self.dtype_)
            idxs = np.arange(chop.len_)
            ts_new[:idx_i] = self.data_[:idx_i]
            ts_new[idx_i:] = \
                t_start + (idxs[idx_i:] - idx_i + 1) * interv + chunk_interv * \
                (idxs[idx_i:, None] >=
                 chop_chunk_edge_idxs[None, idx_chunk_i:]).astype(float). \
                    sum(axis=-1)

        warnings.warn("Times stamp is rebuilt.", UserWarning)
        return TimeStamps(arr_in=ts_new)

    def get_time(self):
        """
        Return an astropy.time.Time object corresponding to the time stamps.

        :return: Time object
        :rtype: astropy.time.core.Time
        """

        return Time(self.data_, format="unix")

    def get_datetime(self):
        """
        Get an array of python datetime.datetime objects corresponding to the
            time stamps, with time zone set to utc.

        :return: array, containing datetime objects
        :rtype: numpy.ndarray
        """

        return self.get_time().to_datetime(timezone=timezone.utc)


class IdArr(DataObj):
    """
    U40 string array recording obs_id
    """

    obj_type_ = "IdArr"  # type: str
    dtype_ = np.dtype("<U40")  # type: numpy.dtype

    def append(self, other):
        """
        Append another IdArr to the end of the current one

        :param IdArr other: IdArr, object to be appended
        """
        return self.append_along_axis(other=other, axis=0)


class ObsInfo(TableObj):
    """
    An object containing observation information in table_, can be initialized
    input table or using read() class method to read in .hk and .run file
    """

    obj_type_ = "ObsInfo"  # type: str

    @classmethod
    def read(cls, filename, try_hk=True, try_run=True):
        """
        Return an ObsInfo object initialized by the housekeeping information
        in .hk and execution information in .run file. Only a selected part of
        runfile information will be added.

        :param str filename: str, path to .hk or .run file or the data file header
        :param bool try_hk: bool, flag whether to try to read .hk file
        :param bool try_run: bool, flag whether to try to read .run file
        :return obs_info: ObsInfo, new object
        :rtype: ObsInfo
        """

        # sections and items to read in .run file
        runfile_keys = {"HEADER": ["RB rc1 data_mode", "RB rc1 servo_mode",
                                   "RB rc2 data_mode", "RB rc2 servo_mode",
                                   "RB rc3 data_mode", "RB rc3 servo_mode",
                                   "RB tes bias"],
                        "FRAMEACQ": True}

        tb_in = Tb(masked=True)
        hk_flag, run_flag = not try_hk, not try_run
        if filename is not None:
            if filename[-3:] == ".hk":
                filename = filename[:-3]
            elif filename[-4:] == ".run":
                filename = filename[:-4]

            if try_run:
                if os.path.isfile(filename + ".hk"):
                    with open(filename + ".hk", "rb") as file:
                        arr_in = np.genfromtxt(file, delimiter=":", comments="#",
                                               dtype=str)
                    for i, (name, val) in enumerate(arr_in):
                        type_flag, name, val = False, name.strip(), val.strip()
                        try:  # try to convert to int
                            val = int(val)
                            type_flag = True
                        except ValueError:
                            pass
                        if not type_flag:  # try to convert to float
                            try:
                                val = float(val)
                                type_flag = True
                            except ValueError:
                                pass
                        if not type_flag:  # try to convert to bool
                            if val.lower() == "true":
                                val = True
                                type_flag = True
                            elif val.lower() == "false":
                                val = False
                                type_flag = True
                        if (not type_flag) and (val.lower().strip() == "none"):
                            continue  # do not add if val is "None"
                        else:
                            tb_in.add_column(
                                    col=Tb.Column(data=[val], name=name))
                    hk_flag = True
                else:
                    warnings.warn("%s.hk does not exist." % filename)

            if try_run:
                if os.path.isfile(filename + ".run"):
                    runfile = MCERunfile(filename=filename + ".run").data
                    for key in runfile_keys:
                        if key in runfile:
                            if runfile_keys[key] is True:  # read all items
                                for item in runfile[key]:
                                    tb_in.add_column(
                                            col=Tb.Column(
                                                    data=[runfile[key][item]],
                                                    name=item.replace(" ", "_")))
                            else:  # read selected items in section
                                for item in runfile_keys[key]:
                                    if item in runfile[key]:
                                        tb_in.add_column(
                                                col=Tb.Column(
                                                        data=[runfile[key][item]],
                                                        name=item.replace(" ", "_")))
                    run_flag = True
                else:
                    warnings.warn("%s.run does not exist." % filename)

            if (not hk_flag) and (not run_flag):
                raise FileNotFoundError("Failed to read info files for %s" %
                                        filename)

        obs_info = cls(tb_in=tb_in)  # initialize object

        return obs_info

    @classmethod
    def read_hk(cls, filename):
        """
        Return an ObsInfo object initialized by the housekeeping information
        in .hk file. Recommended to use ObsInfo.read().

        :param str filename: str, path to .hk file, will try to read filename.hk if
            the first attempt of read in fails
        :return obs_info: ObsInfo, new object
        :rtype: ObsInfo
        """

        tb_in = Tb(masked=True)
        if filename is not None:
            try:
                with open(filename, "rb") as file:
                    arr_in = np.genfromtxt(file, delimiter=":", comments="#",
                                           dtype=str)
            except FileNotFoundError:
                try:
                    with open(filename + ".hk", "rb") as file:
                        arr_in = np.genfromtxt(file, delimiter=":", comments="#",
                                               dtype=str)
                except FileNotFoundError:
                    raise FileNotFoundError("%s or %s.hk does not exist." %
                                            (filename, filename))
            for i, (name, val) in enumerate(arr_in):
                type_flag, name, val = False, name.strip(), val.strip()
                try:  # try to convert to int
                    val = int(val)
                    type_flag = True
                except ValueError:
                    pass
                if not type_flag:  # try to convert to float
                    try:
                        val = float(val)
                        type_flag = True
                    except ValueError:
                        pass
                if not type_flag:  # try to convert to bool
                    if val.lower() == "true":
                        val = True
                        type_flag = True
                    elif val.lower() == "false":
                        val = False
                        type_flag = True
                if (not type_flag) and (val.lower().strip() == "none"):
                    continue  # do not add if val is "None"
                else:
                    tb_in.add_column(col=Tb.Column(data=[val], name=name))
        # assign values
        obs_info = cls(tb_in=tb_in)

        return obs_info


class ObsLog(TableObj):
    """
    An object containing observation logs.
    """

    obj_type_ = "ObsLog"  # type: str

    def __init__(self, tb_in=None):
        """
        Initialize ObsLog object by input table tb_in. Will change 'UTC' column
        to iso format, and change 'shutter closed in' 'mm PWV', 'N/A' in
        'X focus' columns to numpy.nan

        :param tb_in: astropy.table or ObsLog instance
        :type tb_in: astropy.table.Table or ObsLog
        """

        super(ObsLog, self).__init__(tb_in=tb_in)

        # fix "shutter closed" in "mm PWV", "N/A" in "focus"
        for colname in ["mm PWV", "X focus", "Y focus", "Z focus"]:
            if colname in self.colnames_:
                if (self.dtype_[colname] != np.dtype("float")) and \
                        self.len_ > 0:
                    arr_new = np.full(self.len_, fill_value=np.nan,
                                      dtype=float)
                    for i, val in enumerate(self.table_[colname]):
                        try:
                            val = float(val.strip())
                            arr_new[i] = val
                        except ValueError:
                            pass
                    self.table_.replace_column(colname,
                                               Tb.Column(arr_new, name=colname))

    @classmethod
    def read(cls, filename):
        """
        Return an ObsLog by reading from the table in .html file.

        :param filename: str, path to the .html file containing obs_array log,
            returns an empty object if filename is None
        :type filename: str or None
        :return obs_log: ObsLog, new object
        :rtype: ObsLog
        """

        if filename is None:
            tb_in = None
        else:
            tb_in = Tb.read(filename, format="html")
        obs_log = cls(tb_in=tb_in)

        return obs_log

    @classmethod
    def read_folder(cls, folder):
        """
        Return an ObsLog object by reading all the .html files in the folder and
        append them together

        :param str folder: path to the folder
        :return obs_log: ObsLog, new object
        :rtype: ObsLog
        """

        if folder is None:
            obs_log = cls.read(None)
        else:
            f_list, html_list = os.listdir(folder), []
            for filename in f_list:  # get all html files
                if filename[-5:] == ".html":
                    html_list.append(filename)
            if len(html_list) == 0:
                raise ValueError("No html file is find in %s.")

            obs_log = cls()
            for f in html_list:
                obs_log_f = cls.read(filename=os.path.join(folder, f))
                obs_log.append(obs_log_f)

        return obs_log

    def take_by_time(self, time):
        """
        Take the entry of the obs_array log such that the input time is between
        the time in 'UTC' and 'UTC' + 'Scan duration' columns. An empty ObsLog
        will be returned if no entry is found

        :param time: float or str or datetime.datetime or astropy.time.Time,
            float input should be value of time stamp, string input should be
            time in iso format or isot format
        :type time: float or str or datetime.datetime or astropy.time.Time
        :return obs_log_new: the entry of obs_array log selected by time
        :rtype: ObsLog
        """

        if isinstance(time, (int, float, np.integer, np.double)):
            time = Time(time, format="unix")
        elif isinstance(time, str):
            try:
                time = Time(time, format="iso")
            except ValueError:
                try:
                    time = Time(time, format="isot")
                except ValueError:
                    raise ValueError("String format should be ISO or ISOT.")
        time = Time(time)

        if self.empty_flag_:
            warnings.warn("The ObsLog object is empty.", UserWarning)
            return self.copy()
        else:
            obs_t_arr = Time(np.char.replace(self.table_["UTC"], "U", "T"),
                             format="isot")
            obs_dt_arr = self.table_["Scan duration"] * units.s
            flag_arr = (obs_t_arr < time) & (time < obs_t_arr + obs_dt_arr)
            idx, num = np.flatnonzero(flag_arr), np.count_nonzero(flag_arr)
            if num == 0:
                warnings.warn("No entry is found in obs log.", UserWarning)
                tb_cut = None
            else:
                tb_cut = self.table_[idx]
            if num > 1:
                warnings.warn("Multiple entries are found.", UserWarning)
            obs_log_new = self.__class__(tb_in=tb_cut)
            return obs_log_new


# TODO: utc_to_time


class Obs(DataObj):
    """
    Object that host all the data of an observation, including data, chop,
    time stamps, ancillary information like housekeeping or obs_array log.
    """

    obj_type_ = "Obs"  # type: str
    obs_id_ = "0"  # type: str # observation id of this object, default is 0
    obs_id_list_ = []  # type: list[str,]
    obs_id_arr_ = None  # type: IdArr
    chop_ = None  # type: Chop
    ts_ = None  # type: TimeStamps
    t_start_time_ = TimeStamps.t_start_time_  # type: Time
    t_end_time_ = TimeStamps.t_end_time_  # type: Time
    obs_info_ = None  # type: ObsInfo

    def __init__(self, arr_in=None, chop=None, ts=None, obs_info=None,
                 obs_id="0", obs_id_list=None, obs_id_arr=None,
                 t_start_time=None, t_end_time=None):
        """
        Initialize the object with input data array, chop, time stamps,
            obs_array info

        :param arr_in: Obs or DataObj object or array, containing observation
            time stream. Can be 1-d, 2-d or 3-d, the last dimension is the time.
            If Obs object is input, all other input will be ignored, and the
            input Obs will be copied and passed as the new object
        :type arr_in: numpy.ndarray or Obs or DataObj or None
        :param chop: Chop or DataObj or array or list or tuple, 1-d, containing
            chop data
        :type chop: Chop or DataObj or numpy.ndarray or list or tuple or None
        :param ts: TimeStamps or DataObj or array or list or tuple, 1-d
            containing time stamps
        :type ts: TimeStamps or DataObj or numpy.ndarray or list or tuple or
            None
        :param obs_info: ObsInfo or TableObj or astropy.table, containing
            observation information
        :type obs_info: ObsInfo or TableObj or astropy.table.table.Table or
            None
        :param str obs_id: str, less than 40 characters
        :param list obs_id_list: list, of obs_id
        :param obs_id_arr: IdArr, indicating the obs_id of each timestamp
        :type obs_id_arr: IdArr or numpy.ndarray or None
        :param astropy.time.Time t_start_time: Time
        :param astropy.time.Time t_end_time: Time
        """

        super(Obs, self).__init__(arr_in=arr_in)
        if isinstance(arr_in, type(self)) and \
                isinstance(self, type(arr_in)):
            pass
        elif isinstance(arr_in, Obs):
            self.obs_id_ = arr_in.obs_id_
            self.obs_id_list_ = arr_in.obs_id_list_
            self.obs_id_arr_ = arr_in.obs_id_arr_
            self.chop_, self.ts_ = arr_in.chop_, arr_in.ts_
            self.t_start_time_ = arr_in.t_start_time_
            self.t_end_time_ = arr_in.t_end_time_
            self.obs_info_ = arr_in.obs_info_
        else:
            self.obs_id_ = str(obs_id)
            self.obs_id_list_ = [self.obs_id_] if obs_id_list is None else \
                list(obs_id_list)
            self.chop_ = Chop(arr_in=chop)
            self.ts_ = TimeStamps(arr_in=ts)
            self.t_start_time_ = self.ts_.t_start_time_ if \
                (t_start_time is None) or \
                (not isinstance(t_start_time, (datetime, Time))) else \
                t_start_time
            self.t_end_time_ = self.ts_.t_end_time_ if \
                (t_end_time is None) or \
                (not isinstance(t_end_time, (datetime, Time))) else \
                t_end_time
            self.obs_id_arr_ = IdArr(arr_in=[self.obs_id_] * self.len_) if \
                obs_id_arr is None else IdArr(arr_in=obs_id_arr)
            self.update_obs_info(obs_info=obs_info)
            self.__check()

    def __operate__(self, other, operator, r=False):
        obs_new = super(Obs, self).__operate__(
                other=other, operator=operator, r=r)
        if isinstance(other, Obs) and not other.empty_flag_:
            obs_new.obs_id_list_ = np.unique(
                    self.obs_id_list_ + other.obs_id_list_).tolist()
            obs_new.obs_info_.append(other.obs_info_)
            if (not obs_new.obs_info_.empty_flag_) and \
                    ("obs_id" in obs_new.obs_info_.colnames_) and \
                    (obs_new.obs_info_.len_ > 0):
                obs_new.obs_info_.table_ = unique(obs_new.obs_info_.table_,
                                                  keys="obs_id")
        return obs_new

    def replace(self, **kwargs):
        """
        Initialize a new Obs with the input keyword parameters replaced by the
        input values, and use the instance variables of the current object to
        initialize the rest of the parameters

        :return: new object
        :rtype: Obs
        """

        for kw in ["chop", "ts", "obs_info", "obs_id", "obs_id_list",
                   "obs_id_arr", "t_start_time", "t_end_time"]:
            if kw not in kwargs:
                kwargs[kw] = self.__dict__["%s_" % kw]

        return super(Obs, self).replace(**kwargs)

    @classmethod
    def read(cls, filename):
        """
        Read time stream from file and initialize a new Obs object with data

        :param filename: str, path to the MCE time stream data, returns an empty
            object if filename is None
        :type filename: str or None
        :return obs_new: Obs, new object
        :rtype: Obs
        """

        if filename is None:
            arr_in, obs_id = None, None
        else:
            file = SmallMCEFile(filename)  # read MCE data by SmallMCEFile() func
            drs = file.Read(row_col=True)
            arr_in = drs.data.copy()  # 3D array of "raw" MCE data
            obs_id = filename.split("/")[-1].split(".")[0]

        obs_new = cls(arr_in=arr_in, obs_id=obs_id)

        return obs_new

    @classmethod
    def read_header(cls, filename, try_data=True, try_chop=True, try_ts=True,
                    try_hk=True, try_info=True):
        """
        Try to read in time stream, chop (.chop), time stamp (.ts) and
            housekeeping (.hk) files using the file header to initialize

        :param str filename: str, path to the MCE time stream data
        :param bool try_data: bool, flag whether to try to read MCE data file
        :param bool try_chop: bool, flag whether to try to read .chop file
        :param bool try_ts: bool, flag whether to try to read .ts file
        :param bool try_hk: bool, flag whether to try to .hk file, will be
            overridden if try_info==True, kept only for compatibility
        :param bool try_info: bool, flag whether to try to read information in
            .hk and .run file
        :return obs_new: Obs, new object
        :rtype: Obs
        """

        if try_data:
            try:
                obs_new = cls.read(filename=filename)
            except FileNotFoundError:
                warnings.warn("%s not found." % filename)
                obs_id = filename.split("/")[-1].split(".")[0]
                obs_new = cls(obs_id=obs_id)
        else:
            obs_id = filename.split("/")[-1].split(".")[0]
            obs_new = cls(obs_id=obs_id)
        if try_chop:
            try:
                chop = Chop.read(filename=filename + ".chop")
                obs_new.update_chop(chop=chop)
            except FileNotFoundError:
                warnings.warn("%s not found." % (filename + ".chop"))
        if try_ts:
            try:
                ts = TimeStamps.read(filename=filename + ".ts")
                obs_new.update_ts(ts=ts)
            except FileNotFoundError:
                warnings.warn("%s not found." % (filename + ".ts"))
        if try_hk and (not try_info):
            try:
                obs_info = ObsInfo.read_hk(filename=filename + ".hk")
                obs_new.update_obs_info(obs_info=obs_info)
            except FileNotFoundError:
                warnings.warn("%s not found." % (filename + ".hk"))
        if try_info:
            try:
                obs_info = ObsInfo.read(filename=filename, try_hk=True,
                                        try_run=True)
                obs_new.update_obs_info(obs_info=obs_info)
            except FileNotFoundError:
                warnings.warn("%s .hk and .run not found." % filename)

        return obs_new

    def update_data(self, arr_in):
        """
        Update the data_ instance variable for time stream data

        :param arr_in: DataObj or array containing the data
        :type arr_in: DataObj or numpy.ndarray
        """

        super(Obs, self).__fill_values__(arr_in=arr_in)
        self.__check()

    def update_chop(self, chop):
        """
        Update the chop_ instance variable for chop signal

        :param chop: Chop or array, containing the chop data from .chop files
        :type chop: Chop or numpy.ndarray
        """

        chop = Chop(arr_in=chop)
        self.chop_ = chop
        if (not self.ts_.empty_flag_) and (not chop.empty_flag_):
            self.update_ts(ts=self.ts_.corr_by_chop(chop=chop))
        self.__check()

    def update_ts(self, ts):
        """
        Update ts_ instance variable for time stamps

        :param ts: TimeStamps or array, of time stamps from .ts files
        :type ts: TimeStamps or numpy.ndarray
        """

        ts = TimeStamps(arr_in=ts)
        if (not self.chop_.empty_flag_) and (not ts.empty_flag_):
            ts = ts.corr_by_chop(chop=self.chop_)
        self.ts_ = ts
        self.t_start_time_ = self.ts_.t_start_time_
        self.t_end_time_ = self.ts_.t_end_time_
        self.__check()

    def update_obs_info(self, obs_info):
        """
        Update obs_info_ instance variable

        :param obs_info: ObsInfo or astropy.table, containing housekeeping info
            in .hk files
        :type obs_info: ObsInfo or astropy.table.table.Table
        """

        obs_info = ObsInfo(tb_in=obs_info)
        obs_info.add_id(obs_id=self.obs_id_)
        self.obs_info_ = obs_info

    def __check(self):
        """
        Check the consistency of length of data_, chop_ and ts_ and update
            obs_id_arr_ according to the length of non-empty instance variable
        :raises ValueError: chop len disagree with data
        """

        if not self.empty_flag_:  # if data_ is not empty
            if self.obs_id_arr_.len_ != self.len_:
                self.obs_id_arr_ = IdArr(arr_in=[self.obs_id_] * self.len_)
            if not self.chop_.empty_flag_:
                if self.len_ != self.chop_.len_:
                    warnings.warn("%s data_ and chop__ length disagree!" %
                                  self.obs_id_)
                    if (self.len_ > 1) and (self.chop_.len_ > 1):
                        raise ValueError(("%s data_ and chop_ " % self.obs_id_) +
                                         "length disagree with both length > 1!")
            elif not self.ts_.empty_flag_:
                if self.len_ != self.ts_.len_:
                    warnings.warn("%s data_ and ts_ length disagree!" %
                                  self.obs_id_)
        elif not self.chop_.empty_flag_:
            if self.obs_id_arr_.len_ != self.chop_.len_:
                self.obs_id_arr_ = IdArr([self.obs_id_] * self.chop_.len_)
            if not self.ts_.empty_flag_:
                if self.chop_.len_ != self.ts_.len_:
                    warnings.warn(("%s chop_ and ts_ length " % self.obs_id_) +
                                  "disagree! Using chop_.")
        else:
            if not self.ts_.empty_flag_:
                if self.obs_id_arr_.len_ != self.ts_.len_:
                    self.obs_id_arr_ = IdArr([self.obs_id_] * self.ts_.len_)
            else:
                self.obs_id_arr_ = IdArr([])

    def __repr__(self):
        return "obs_id: %s, start: %s, end: %s" % \
               (self.obs_id_, self.t_start_time_.to_value(format="iso"),
                self.t_end_time_.to_value(format="iso")) + "\n\t" + \
               super(Obs, self).__repr__() + "\n\t" + \
               self.chop_.__repr__() + "\n\t" + self.ts_.__repr__() + "\n\t" + \
               self.obs_info_.__repr__()

    def __eq__(self, other):
        same_flag = super(Obs, self).__eq__(other)
        if same_flag:
            same_flag = same_flag & (self.chop_.__eq__(other.chop_))
            same_flag = same_flag & (self.ts_.__eq__(other.ts_))
            same_flag = same_flag & (self.obs_id_arr_.__eq__(other.obs_id_arr_))

        return same_flag

    def to_obs_array(self, array_map=None):
        """
        Re-order the data_ according to the input array_map, and create an
        ObsArray object to return. The data_ must have at least 2 dimensions,
        the first axis as mce row and the second as mce column. If array_map is
        left None, will create an array map that maps the first dimension(row)
        to spat, the second dimension(col) to spec

        :param array_map: ArrayMap or list or tuple or array, to initialize
            ArrayMap object
        :type array_map: ArrayMap or list or tuple or numpy.ndarray or None
        :return obs_array: ObsArray object
        :rtype ObsArray: ObsArray
        :raises ValueError: not enough dim, data_ shape smaller than array map
        """

        # TODO: check whether ts is initialized before conversion
        array_map = ArrayMap(arr_in=array_map)
        if array_map.empty_flag_ and self.empty_flag_:
            data_sorted = np.empty(0, dtype=self.dtype_)
        else:
            data_use = self.data_
            if array_map.empty_flag_:
                data_use = self.data_[None, ...] if self.ndim_ < 2 else \
                    self.data_
                col_idxs, row_idxs = np.meshgrid(range(self.shape_[1]),
                                                 range(self.shape_[0]))
                col_idxs, row_idxs = col_idxs.flatten(), row_idxs.flatten()
                arr_map = np.array([row_idxs, col_idxs, row_idxs, col_idxs]). \
                    transpose()
                array_map = ArrayMap(arr_in=arr_map)
            elif data_use.ndim < 2:
                raise ValueError("Not enough dimensions.")
            shape_mce = data_use.shape[:2]
            if (array_map.mce_row_ulim_ >= shape_mce[0]) or \
                    (array_map.mce_col_llim_ >= shape_mce[1]):
                raise ValueError("Data shape %s " % str(self.shape_) +
                                 "inconsistent with array map " +
                                 "row: %i col: %i." % (array_map.mce_row_ulim_,
                                                       array_map.mce_col_ulim_))
            # get indices of the flattened first 2 dimensions
            new_shape = (np.prod(shape_mce),) + data_use.shape[2:]
            idx_arr = np.arange(int(np.prod(shape_mce))).reshape(shape_mce)
            data_sorted = data_use.reshape(new_shape)[
                [idx_arr[tuple(idx)] for idx in array_map.mce_idxs_]]

        obs_array = ObsArray(
                arr_in=data_sorted, array_map=array_map, chop=self.chop_,
                ts=self.ts_, obs_info=self.obs_info_, obs_id=self.obs_id_,
                obs_id_list=self.obs_id_list_, obs_id_arr=self.obs_id_arr_,
                t_start_time=self.t_start_time_,
                t_end_time=self.t_end_time_)

        return obs_array

    def append(self, other):
        """
        Append another obs_array object to the current one along the time axis,
        including data_, chop_, ts_, obs_info_, obs_id_arr_ and obs_id_list_.

        :param Obs other: Obs, object to append
        """
        if not isinstance(other, Obs):  # check type
            raise TypeError("Invalid type of input, expect Obs")

        if (not other.empty_flag_) and self.empty_flag_:
            self.obs_id_ = other.obs_id_
            self.obs_id_list_ = other.obs_id_list_
        else:
            for obs_id in other.obs_id_list_:
                if obs_id not in self.obs_id_list_:
                    self.obs_id_list_.append(obs_id)
        super(Obs, self).append_along_axis(other=other, axis=-1)
        if (not other.ts_.empty_flag_) and \
                (self.ts_.empty_flag_ or
                 ((not self.ts_.empty_flag_) and
                  (self.t_start_time_ > other.t_start_time_))):
            self.t_start_time_ = other.t_start_time_
        if (not other.ts_.empty_flag_) and \
                (self.ts_.empty_flag_ or
                 ((not self.ts_.empty_flag_) and
                  (self.t_end_time_ < other.t_end_time_))):
            self.t_end_time_ = other.t_end_time_
        self.chop_.append(other.chop_)
        self.ts_.append(other.ts_)
        self.obs_info_.append(other.obs_info_)
        self.obs_id_arr_.append(other.obs_id_arr_)

        self.__check()

    def expand(self, other, axis=0):
        """
        Expand the obs_array.data_ by appending data_ other of another Obs object at
        the given axis in array, axis can not be the last axis

        :param Obs other: Obs, the other object used to expand the data_, should
            have the same chop_, ts_, obs_id_, obs_id_arr_
        :param int axis: int, the axis to expand data_, can not be -1 or the
            last axis
        :raises TypeError: invalid input type
        :raises ValueError: invalid axis, chop_, ts_, obs_id_, obs_id_arr_ do
            not agree
        """

        if not isinstance(other, Obs):  # check type
            raise TypeError("Invalid type of input, expect Obs")

        axis = self.__check_axis__(axis=axis)
        if (not self.empty_flag_) and (not self.ts_.empty_flag_) and \
                axis == self.ndim_ - 1:
            self.append(other=other)
        else:
            empty_flag = self.empty_flag_
            if (not self.chop_.empty_flag_) and \
                    (not other.chop_.empty_flag_) and \
                    (self.chop_ != other.chop_):
                raise ValueError("chop_ do not agree")
            if (not self.ts_.empty_flag_) and (not other.ts_.empty_flag_) and \
                    self.ts_ != other.ts_:
                raise ValueError("ts_ do not agree")
            if (not self.obs_id_arr_.empty_flag_) and \
                    (not other.obs_id_arr_.empty_flag_) and \
                    self.obs_id_arr_ != other.obs_id_arr_:
                raise ValueError("obs_id_arr_ do not agree")

            super(Obs, self).append_along_axis(other=other, axis=axis)
            if empty_flag and (not other.empty_flag_):
                self.update_chop(other.chop_)
                self.update_ts(other.ts_)
                self.obs_id_ = other.obs_id_
                self.obs_id_list_ = other.obs_id_list_
                self.obs_id_arr_ = other.obs_id_arr_
                self.update_obs_info(other.obs_info_)

    def proc_along_axis(self, method="nanmean", axis=0, **kwargs):
        """
        Process data_ using the method in the specified axis. returns a new
        Object with the same ndim_, but length 1 in the given axis. If the last
        axis is specified, will call proc_along_time().

        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'nanmad'
            'mean', 'median', 'sum', 'std', 'min', 'max', 'mad',
            'num', 'num_not_is_nan', 'num_is_finite', the method to calculate
            data value in each chunk
        :param int axis: int, axis index to process the data, allow range is
            -ndim_ to ndim_-1
        :param kwargs: keyword arguments to initialize a new object
        :return obs_proc: Obs, new object
        :rtype: Obs
        """

        axis = self.__check_axis__(axis=axis)
        if axis == self.ndim_ - 1:
            obs_proc = self.proc_along_time(method=method, **kwargs)
        else:
            obs_proc = super(Obs, self).proc_along_axis(
                    method=method, axis=axis, **kwargs)

        return obs_proc

    def proc_along_time(self, method="nanmean", **kwargs):
        """
        Process data_ using the method in the last axis (time axis). returns a
        new Object with the same ndim_, but length 1 in the last axis, and chop_
        ts_ will be averaged by nanmean, obs_id_arr_ will use the obs_id of the
        object

        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'nanmad'
            'mean', 'median', 'sum', 'std', 'min', 'max', 'mad',
            'num', 'num_not_is_nan', 'num_is_finite', the method to calculate
            data value in each chunk
        :param kwargs: keyword arguments of other parameters to replace in the
            returned object
        :return Obs obs_new: Obs, new object
        """

        chop_new = self.chop_.proc_along_axis(method="nanmean", axis=-1) \
            if not self.chop_.empty_flag_ else self.chop_
        ts_new = self.ts_.proc_along_axis(method="nanmean", axis=-1) \
            if not self.ts_.empty_flag_ else self.ts_
        obs_id_arr_new = IdArr(arr_in=[self.obs_id_]) \
            if not self.obs_id_arr_.empty_flag_ else self.obs_id_arr_
        obs_new = super(Obs, self).proc_along_axis(
                method=method, axis=-1, chop=chop_new, ts=ts_new,
                obs_info=self.obs_info_, obs_id_arr=obs_id_arr_new, **kwargs)

        return obs_new

    def take_by_flag_along_axis(self, flag_arr, axis=0, **kwargs):
        """
        Create a new object from the current Obs whose data are selected by
        the bool array flag in the axis specified.

        :param flag_arr: bool array, 1-d, the length should match with
            self.len_, the data flagged as True will be used in the new
            Obs
        :type flag_arr: list or tuple or numpy.ndarray
        :param int axis: int, the index of axis, if the last axis is used, will
            call take_by_flag_along_time()
        :param kwargs: keyword arguments to initialize a new object
        :return obs_new: Obs, a new object containing the data flagged
        :rtype: Obs or child class
        """

        axis = self.__check_axis__(axis=axis)
        if axis == self.ndim_ - 1:
            obs_new = self.take_by_flag_along_time(flag_arr=flag_arr, **kwargs)
        else:
            obs_new = super(Obs, self).take_by_flag_along_axis(
                    flag_arr=flag_arr, axis=axis, **kwargs)

        return obs_new

    def take_by_flag_along_time(self, flag_arr=None, chop=True, **kwargs):
        """
        Create a new object from the current Obs whose data are selected by
        the bool array flag in the last axis.

        :param flag_arr: bool array, 1-d, the length should match with
            self.len_, the data flagged as True will be used in the new
            Obs
        :type flag_arr: list or tuple or numpy.ndarray matching the last axis of
            data_; if left None, will try to take data by chop phase; if the
            input condition is all False, will return an empty object
        :param bool chop: bool, the chop phase to take, if flag_arr is left None
            and chop is True, on chop data with chop_.data_=True will be taken,
            otherwise off chop data will be taken
        :param kwargs: keyword arguments to initialize a new object
        :return obs_new: Obs, a new object containing the data flagged
        :rtype: Obs or child class
        :raises ValueError: length mismatch
        """

        if flag_arr is None:
            if self.chop_.len_ != self.len_:
                raise ValueError("Chop length disagree with data length.")
            flag_arr = (self.chop_.data_ == chop)

        if np.any(flag_arr):
            chop_new = self.chop_.take_by_flag_along_axis(flag_arr, axis=-1) \
                if not self.chop_.empty_flag_ else self.chop_
            ts_new = self.ts_.take_by_flag_along_axis(flag_arr, axis=-1) \
                if not self.ts_.empty_flag_ else self.ts_
            obs_id_arr_new = self.obs_id_arr_.take_by_flag_along_axis(
                    flag_arr, axis=-1) \
                if not self.obs_id_arr_.empty_flag_ else self.obs_id_arr_
            obs_id_list_new = np.unique(obs_id_arr_new.data_).tolist()
            obs_id_new = obs_id_list_new[0] if len(obs_id_list_new) > 0 else "0"
            if not self.obs_info_.empty_flag_ and \
                    "obs_id" in self.obs_info_.colnames_:
                tb_flag = np.any([self.obs_info_.table_["obs_id"] == obs_id
                                  for obs_id in obs_id_list_new], axis=0)
                obs_info_new = ObsInfo(tb_in=self.obs_info_.table_[tb_flag])
            else:
                obs_info_new = self.obs_info_
            obs_new = super(Obs, self).take_by_flag_along_axis(
                    flag_arr=flag_arr, axis=-1, chop=chop_new, ts=ts_new,
                    obs_info=obs_info_new, obs_id=obs_id_new,
                    obs_id_list=obs_id_list_new, obs_id_arr=obs_id_arr_new,
                    t_start_time=None, t_end_time=None, **kwargs)
        else:
            obs_new = self.__class__()

        return obs_new

    def take_by_idx_along_time(self, idxs, **kwargs):
        """
        Create a new object from the current Obs whose data are selected by
        the indices in idxs along the time axis. Call take_by_flag_along_axis(),
        so the returned object may not have the same order as in idxs.

        :param idxs: int or list or tuple or numpy.ndarray, value or list of
            value of indices to take
        :type idxs: int or list or tuple or numpy.ndarray
        :param kwargs: keyword arguments to initialize a new object, for
            backward compatibility
        :return data_new: DataObj, a new object containing the data flagged
        :rtype: DataObj or child class
        """
        return self.take_by_idx_along_axis(idxs=idxs, axis=-1, **kwargs)

    def take_by_obs_id(self, obs_id, **kwargs):
        """
        Create a new object from the part of the current DataObj that
        obs_id_arr_ matches with the input obs_id

        :param str obs_id: str, the obs_id of the obs_array to be taken. If obs_id is
            not found in obs_id_arr_, an empty Obs will be returned
        :return: new object containing the part
        :rtype: Obs
        """

        flag_arr = (self.obs_id_arr_.data_ == obs_id)
        if np.count_nonzero(flag_arr) == 0:
            warnings.warn("%s is not found in obs_array." % obs_id)

        return self.take_by_flag_along_time(flag_arr=flag_arr, **kwargs)

    def take_when(self, t_ran, **kwargs):
        """
        Take the part of the data with t_ran[0] <= ts <= t_ran[1]

        :param t_ran: tuple or list or array, (min_time_stamp, max_time_stamp)
        :type t_ran: tuple or list or numpy.ndarray
        """

        flag_arr = (self.ts_.data_ >= t_ran[0]) & (self.ts_.data_ <= t_ran[1])

        return self.take_by_flag_along_time(flag_arr=flag_arr, **kwargs)

    def resample_by_ts(self, ts_new, method="interpolation",
                       fill_value=np.nan, **kwargs):
        """
        Project the data_ to new time stamps by either interpolate data_ by the
        new time stamps or pick the values whose time stamps match exactly. If
        method is 'interpolation', the method will call tools.interp_nan, and
        for pixels with all nan value, they will be filled with input
        fill_value. If the method is 'exact', the data points whose time stamps
        don't exist in the original time stamps will be filled with fill_value.
        The new Obs object will lose chop_, obs_info_ information.

        :param TimeStamps ts_new: TimeStamps, recording the new time stamps to
            use
        :param str method: str, method to project data, allowed values are
            'interpolation' and 'exact'
        :param fill_value: the value to fill in the case that all vales are
            non-finite values for interpolation, or some time stamps don't
            exist in the original one for 'exact' method
        :param kwargs: keyword arguments of other parameters to replace in the
            object returned
        :return: new Obs object
        :rtype: Obs
        """

        ts_new = TimeStamps(ts_new)
        if self.ts_.empty_flag_:
            raise ValueError("ts_ of the current object is not initialized.")

        if method.lower().strip()[0] == "i":  # for interpolation
            interp_func = lambda arr: naninterp(
                    x=ts_new.data_, xp=self.ts_.data_, fp=arr,
                    fill_value=fill_value)
            data_new = np.apply_along_axis(interp_func, axis=-1, arr=self.data_)
        elif method.lower().strip()[0] == "e":  # for exact
            new_shape = self.shape_[:-1] + (ts_new.len_,)
            data_new = np.full(new_shape, fill_value=fill_value,
                               dtype=self.dtype_)
            idx, idx_new = [], []
            for i in range(ts_new.len_):
                idx_found = np.flatnonzero(
                        (self.ts_.data_ - ts_new.data_[i] == 0))
                if len(idx_found) > 0:
                    idx.append(idx_found[0])
                    idx_new.append(i)
            data_new[..., idx_new] = self.data_[..., idx]
        else:
            raise ValueError("Invalid input for method.")

        return self.replace(arr_in=data_new, chop=None, ts=ts_new,
                            obs_id_arr=None, t_start_time=None, t_end_time=None,
                            **kwargs)

    def chunk_split(self, chunk_edge_idxs=None):
        """
        Chunk the current data in the last axis according to the edge indices in
        chunk_edge_idxs, put each chunk into a new Obs, return the list of
        the Obs.

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func. If left None, will use
            chop_.chunk_edge_idxs_
        :return obs_chunk_list: list, containing Obs objects of data chunks
            split by chunk_edge_idxs
        :rtype: list
        """

        if chunk_edge_idxs is None:
            chunk_edge_idxs = self.chop_.chunk_edge_idxs_
        chunk_num = len(chunk_edge_idxs) - 1
        obs_chunk_list = super(Obs, self).chunk_split(chunk_edge_idxs)
        chop_chunk_list = self.chop_.chunk_split(chunk_edge_idxs) \
            if not self.chop_.empty_flag_ else [None] * chunk_num
        ts_chunk_list = self.ts_.chunk_split(chunk_edge_idxs) \
            if not self.ts_.empty_flag_ else [None] * chunk_num
        obs_id_arr_chunk_list = self.obs_id_arr_.chunk_split(chunk_edge_idxs)
        for (obs, chop, ts, obs_id_arr) in \
                zip(obs_chunk_list, chop_chunk_list, ts_chunk_list,
                    obs_id_arr_chunk_list):
            obs.update_chop(chop=chop)
            obs.update_ts(ts=ts)
            obs.obs_id_arr_ = obs_id_arr
            obs_id_list = np.unique(obs_id_arr.data_).tolist()
            obs.obs_id_, obs.obs_id_list_ = obs_id_list[0], obs_id_list
            if "obs_id" in self.obs_info_.colnames_:
                tb_flag = np.any([
                    self.obs_info_.table_["obs_id"] == obs_id
                    for obs_id in obs_id_list], axis=1)
                obs_info = ObsInfo(tb_in=self.obs_info_.table_[tb_flag])
            else:
                obs_info = self.obs_info_
            obs.update_obs_info(obs_info=obs_info)

        return obs_chunk_list

    def chunk_proc(self, method="nanmean", chunk_edge_idxs=None,
                   keep_shape=False, **kwargs):
        """
        Chunk the current data in the last axis according to the edge indices in
        chunk_edge_idxs, and process data_ by mean, median, sum etc. in each
        chunk, and return a new Obs object with processed chunk data. The data
        in chop_ and ts_ will be chunk-processed with nanmean, and for
        obs_id_arr_, the values at the indices in chunk_edge_idxs will be used.

        :param numpy.ndarray chunk_edge_idxs: array, index of chunk of each data
            point, out put of index_diff_edge() func. If left None, will use
            chop_.chunk_edge_idxs_
        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'mean',
            'median', 'sum', 'std', 'min', 'max', 'num', 'num_not_is_nan',
            'num_is_finite', the method to calculate data value in each chunk
        :param keep_shape: bool, flag whether the output should have the shape
            as the input, so that each chunk are not compressed
        :param kwargs: keyword arguments to initialize a new object
        :return obs_chunked: Obs, dtype_ will be np.double for calculations
            or int for counting numbers, contained chunked data
        :rtype: Obs
        """

        if chunk_edge_idxs is None:
            chunk_edge_idxs = self.chop_.chunk_edge_idxs_
        # check method value
        chop_chunked = self.chop_.chunk_proc(
                chunk_edge_idxs=chunk_edge_idxs, method="nanmean",
                keep_shape=keep_shape) if not self.chop_.empty_flag_ else None
        ts_chunked = self.ts_.chunk_proc(
                chunk_edge_idxs=chunk_edge_idxs, method="nanmean",
                keep_shape=keep_shape) if not self.ts_.empty_flag_ else None
        if keep_shape:
            obs_id_arr_chunked = self.obs_id_arr_
        else:
            obs_id_arr_chunked = self.obs_id_arr_.take_by_idx_along_axis(
                    chunk_edge_idxs[:-1])
        obs_chunked = super(Obs, self).chunk_proc(
                method=method, chunk_edge_idxs=chunk_edge_idxs,
                keep_shape=keep_shape, chop=chop_chunked, ts=ts_chunked,
                obs_id_arr=obs_id_arr_chunked, **kwargs)

        return obs_chunked

    def flatten(self):
        """
        Return a new object with all the axis except the time axis removed. If
        ts_ or chop_ or obs_id_arr_ is initialized, the result will be 2-d with
        shape (-1, len_), otherwise 1-d.
        """

        if (not self.ts_.empty_flag_) or (not self.chop_.empty_flag_) or \
                (not self.obs_id_arr_.empty_flag_):
            return self.replace(arr_in=self.data_.reshape(-1, self.len_))
        else:
            return self.replace(arr_in=self.data_.flatten())

    def get_chop_freq(self):
        """
        Calculate chop frequency by calculating 1 / the median value of time
        interval between on-chops; returns 0 if there is only 1 chop.

        :return: chop frequency
        :rtype: float
        raises ValueError: ts_ or chop_ not initialized
        """

        if self.chop_.empty_flag_ or self.ts_.empty_flag_:
            raise ValueError("Need to initialize chop_ and ts_ first.")

        on_chop_ts = self.ts_.take_by_flag_along_axis(flag_arr=self.chop_.data_)
        on_chop_chop = self.chop_.take_by_flag(flag_arr=self.chop_.data_)
        if on_chop_chop.len_ == 0:
            return 0

        chunk_ts = on_chop_ts.chunk_proc(on_chop_chop.chunk_edge_idxs_,
                                         method="nanmean")
        if chunk_ts.len_ == 1:
            return 0

        med_interv = np.nanmedian(np.diff(chunk_ts.data_))
        if np.isfinite(med_interv):
            return 1 / med_interv
        else:
            return 0

    def match_obs_log(self, obs_log, time_offset=30):
        """
        find the entry in obs_log matching the time of the object, and add the
        entry in obs_info_, only update the object with no return

        :param ObsLog obs_log: ObsLog, recording the observation log
        :param time_offset: int or float, time offset in second used in
            searching in obs_log, a positive number means the data is delayed
            against obs log entry
        :type time_offset: int or float
        """

        obs_log = ObsLog(obs_log)
        entries = obs_log.take_by_time(time=self.t_start_time_ + (
                self.t_end_time_ - self.t_start_time_) / 2 -
                                            time_offset * units.s)
        self.obs_info_.expand(entries)


# TODO: to_table
# TODO: def bin(self, method):


class ObsArray(Obs):
    """
    Object that can handle data with array map information
    """

    obj_type_ = "ObsArray"  # type: str
    array_map_ = None  # type: ArrayMap

    def __init__(self, arr_in=None, array_map=None, **kwargs):
        """
        Initialize the object with input data array, chop, time stamps,
        obs_array info

        :param array_map: ArrayMap or array, to initialize ArrayMap, if left
            None, [[0,0,0,0], [1,0,1,0], ...] will be used
        :type array_map: ArrayMap or numpy.ndarray
        :param kwargs: keyword arguments passed to Obs()
        """

        super(ObsArray, self).__init__(arr_in=arr_in, **kwargs)
        if isinstance(arr_in, type(self)) and isinstance(self, type(arr_in)):
            pass
        elif isinstance(arr_in, ObsArray):
            self.array_map_ = arr_in.array_map_
        else:
            if (self.ndim_ > 2) and (array_map is None):
                super(ObsArray, self).__init__(arr_in=Obs(
                        arr_in=arr_in, **kwargs).to_obs_array())
            else:
                if array_map is None:
                    array_map_list = [np.arange(self.shape_[0]).tolist(),
                                      [0] * self.shape_[0]] * 2
                    array_map = np.array(array_map_list).transpose()
                self.update_array_map(array_map=array_map)

    def replace(self, **kwargs):
        """
        Initialize a new ObsArray with the input keyword parameters replaced by
        the input values, and use the instance variables of the current object
        to initialize the rest of the parameters

        :return: new object
        :rtype: ObsArray
        """

        for kw in ["array_map"]:
            if kw not in kwargs:
                kwargs[kw] = self.__dict__["%s_" % kw]

        return super(ObsArray, self).replace(**kwargs)

    @classmethod
    def from_table(cls, tb_in):
        """
        Generate an ObsArray object from the table generated by the to_table
            method

        :param astropy.table.Table tb_in: input table
        :return: ObsArray object containing data in the table
        :rtype: ObsArray
        """

        tb_in = Tb(tb_in, masked=True)
        if len(tb_in) == 0:
            return cls()

        colnames = np.array(tb_in.colnames)
        colnames_lower = np.char.lower(colnames)
        if "spatial_position" in colnames_lower:  # check main column
            horizontal = True
        elif "spectral_index" in colnames_lower:
            horizontal = False
        else:
            raise ValueError("Unsupported table format, missing " +
                             "spatial_position or spectral_position column.")
        colname_main, colname_repeat = ("spatial_position", "spectral_index")[
                                       ::(1 if horizontal else -1)]
        colname_main = colnames[
            np.argwhere(colnames_lower == colname_main)[0, 0]]
        row_data = tb_in[colname_main].data.astype(int).filled(-1)
        row_arr, row_counts = np.unique(row_data, return_counts=True)
        row_num = len(row_arr)

        col_list, col_data_list = [], []  # prepare data in the repeated columns
        for colname in colnames:
            if colname_repeat in colname.lower():
                col_list.append(int(colname.split("=")[-1]))
                col_data_list.append(tb_in[colname].data)
        if len(col_list) == 0:
            raise ValueError("Unsupported table format, missing " +
                             "the repeated columns for the other dimension")
        col_data = np.ma.masked_array(col_data_list).transpose()
        col_arr = np.array(col_list, dtype=int)
        col_num = len(col_arr)

        col_time, data_time_list = [], []  # look for column in the time axis
        if "time_stamp" in colnames_lower:
            colname_time = colnames[
                np.argwhere(colnames_lower == "time_stamp")[0, 0]]
            col_time.append("time_stamp")
            data_time_list.append(
                    tb_in[colname_time].data.astype(TimeStamps.dtype_).filled(0))
        if "obs_id" in colnames_lower:
            colname_time = colnames[
                np.argwhere(colnames_lower == "obs_id")[0, 0]]
            col_time.append("obs_id")
            data_time_list.append(
                    tb_in[colname_time].data.astype(IdArr.dtype_).filled("0"))
        if len(colname_time) == 0:
            if row_counts.max() > 1:
                warnings.warn("Missing time_stamp and obs_id with ambiguous " +
                              "time sequence.", UserWarning)
                col_time.append("time_stamp")
                data_time_list.append(np.array(
                        [np.count_nonzero(row_data[:i] == val for i, val in
                                          enumerate(row_data))]))
        if "chop_phase" in colnames_lower:
            colname_time = colnames[
                np.argwhere(colnames_lower == "chop_phase")[0, 0]]
            col_time.append("chop_phase")
            if isinstance(tb_in[colname_time][0],
                          (int, float, bool, np.integer, np.double)):
                chop_data = \
                    tb_in[colname_time].data.astype(Chop.dtype_).filled(False)
            elif isinstance(tb_in[colname_time][0], str):
                chop_data = \
                    (np.char.lower(tb_in[colname_time].filled("False")) ==
                     "true") | \
                    (np.char.lower(tb_in[colname_time].filled("F")) == "t")
            else:
                warnings.warn("Data type of chop_phase is not understood.",
                              UserWarning)
                chop_data = tb_in[colname_time].data.filled()
            data_time_list.append(chop_data)
        col_time = np.array(col_time)
        time_arr = np.unique(data_time_list[0])
        time_num = np.unique(data_time_list, axis=-1).shape[-1]
        if np.any(len(time_arr) != time_num):
            raise ValueError("time sequence column is not unique.")

        # build array to store data, and turn into ObsArray format
        data_new = np.ma.empty((row_num, col_num, time_num), dtype=np.double)
        data_new.masked = True
        row_idx_dict = {row: i for (i, row) in enumerate(row_arr)}
        time_idx_dict = {time: i for (i, time) in enumerate(time_arr)}
        row_idx = np.array([row_idx_dict[row] for row in row_data])
        time_idx = np.array([time_idx_dict[time] for
                             time in data_time_list[0]])
        data_new[row_idx, :, time_idx] = col_data
        data_flattened = data_new.reshape(-1, time_num)
        array_map_new = np.array(
                np.meshgrid(col_arr, row_arr)[::-1 if horizontal else 1] +
                np.meshgrid(np.arange(col_num), np.arange(row_num))[::-1]). \
            reshape(4, -1).transpose()
        flag_use = (np.ma.count_masked(data_flattened, axis=-1) < time_num)
        data_use = data_flattened[flag_use].filled(fill_value=np.nan)
        array_map_use = array_map_new[flag_use]

        time_idx_arr = np.array([np.argwhere(data_time_list[0] == time)[0, 0]
                                 for time in time_arr])
        time_data_list = []  # get ts, chop, obs_id
        for colname in ("time_stamp", "obs_id", "chop_phase"):
            if colname in col_time:
                col_time_idx = np.argwhere(col_time == colname)[0, 0]
                time_data_list.append(
                        data_time_list[col_time_idx][time_idx_arr])
            else:
                time_data_list.append(None)
        obs_id, obs_id_list = ("0", None) if time_data_list[1] is None \
            else (time_data_list[1][0], time_data_list[1].tolist())
        obs_info = Tb()  # get obs_info
        for colname in colnames:
            if ~np.any([name in colname.lower() for name in
                        (colname_main, colname_repeat,
                         "time_stamp", "chop_phase")]):
                obs_info.add_column(tb_in[colname])
        obs_info = unique(obs_info) if len(obs_info) > 0 else None

        obs_array_new = cls(arr_in=data_use, array_map=array_map_use,
                            chop=time_data_list[2], ts=time_data_list[0],
                            obs_info=obs_info, obs_id=obs_id,
                            obs_id_list=obs_id_list,
                            obs_id_arr=time_data_list[1])

        return obs_array_new

    @classmethod
    def read_table(cls, filename, *args, **kwargs):
        """
        read data from a table with Astropy.table.Table.read method, and
        generate an ObsArray object using ObsArray.from_table method

        :para str filename: path to the table file
        :para args: arguments to pass to Astropy.table.Table.read
        :para kwargs: keyword arguments to pass to Astropy.table.Table.read,
            e.g. format='csv'
        :return: ObsArray object containing data in the table
        :rtype: ObsArray
        """

        tb_in = Tb.read(filename, *args, **kwargs)
        obs_array_new = cls.from_table(tb_in)

        return obs_array_new

    def update_data(self, arr_in):
        """
        Update data_ instance

        :param numpy.ndarray arr_in: array, data to use
        """

        super(ObsArray, self).update_data(arr_in=arr_in)
        self.__check()

    def update_array_map(self, array_map):
        """
        Update array_map_ instance variable for time stamps

        :param array_map: ArrayMap or array, to initialize ArrayMap object
        :type array_map: ArrayMap or numpy.ndarray
        """

        array_map = ArrayMap(arr_in=array_map)
        self.array_map_ = array_map
        self.__check()

    def __check(self):
        """
        Call Obs.__check(), then check dimension of data and length of array_map
        """

        super(ObsArray, self)._Obs__check()
        if self.array_map_.len_ != self.shape_[0]:
            raise ValueError(("%s inconsistent length " % self.obs_id_) +
                             "between data and array map.")

    def __repr__(self):
        return super(ObsArray, self).__repr__() + "\n\t" + \
               self.array_map_.__repr__()

    def __eq__(self, other):
        same_flag = super(ObsArray, self).__eq__(other)
        if same_flag:
            same_flag = same_flag & \
                        (self.array_map_.__eq__(other.array_map_))

        return same_flag

    def append(self, other):
        """
        Same as Obs.append(), also compares array_map_, and will updates
        array_map_ if the current object is empty

        :param ObsArray other: ObsArray, object to append
        :raises ValueError: array map mismatch
        """

        if not isinstance(other, ObsArray):  # check type
            raise TypeError("Invalid type of input, expect ObsArray")

        if other.empty_flag_:
            pass
        else:
            if not self.empty_flag_:
                if self.array_map_ != other.array_map_:
                    raise ValueError("Array map mismatch.")
            super(ObsArray, self).append(other=other)
            self.update_array_map(array_map=other.array_map_)
            self.__check()

    def expand(self, other):
        """
        Expand the data_ by appending data_ other of another ObsArray object at
        after the current object, and append their array map

        :param ObsArray other: ObsArray, the other object used to expand the
            data_, should have the same chop_, ts_, obs_id_, obs_id_arr_
        """

        super(ObsArray, self).expand(other=other, axis=0)
        self.array_map_.expand(other.array_map_)

    def proc_along_axis(self, method="nanmean", axis=0, **kwargs):
        """
        Process data_ using the method in the specified axis. returns a new
        Object with the same ndim_, but length 1 in the given axis. If the last
        axis is specified, will call proc_along_time(), if the first axis is
        used, the array_map_ information will be set to default

        :param str method: str, default 'nanmean', allowed values are 'nanmean',
            'nanmedian', 'nansum', 'nanstd', 'nanmin', 'nanmax', 'nanmad'
            'mean', 'median', 'sum', 'std', 'min', 'max', 'mad',
            'num', 'num_not_is_nan', 'num_is_finite', the method to calculate
            data value in each chunk
        :param int axis: int, axis index to process the data, allow range is
            -ndim_ to ndim_-1
        :param kwargs: keyword arguments to initialize a new object
        :return obs_array_proc: ObsArray, new object
        :rtype: ObsArray
        """

        axis = self.__check_axis__(axis=axis)
        if axis == 0:
            if "array_map" not in kwargs:
                kwargs["array_map"] = None
            obs_array_proc = super(Obs, self).proc_along_axis(
                    method=method, axis=axis, **kwargs)
        elif axis == self.ndim_ - 1:
            obs_array_proc = self.proc_along_time(method=method, **kwargs)
        else:
            obs_array_proc = super(Obs, self).proc_along_axis(
                    method=method, axis=axis, array_map=self.array_map_,
                    **kwargs)

        return obs_array_proc

    def proc_along_time(self, *args, **kwargs):
        """
        Same as Obs.proc_along_time()

        :return: ObsArray, new object
        :rtype: ObsArray
        """
        return super(ObsArray, self).proc_along_time(
                *args, **kwargs, array_map=self.array_map_)

    def take_by_flag_along_axis(self, flag_arr, axis=0, **kwargs):
        """
        Take part of the data by flag, if axis = 0, will also cut array_map_, if
        axis is the last axis, will call take_by_flag_along_time()

        :param numpy.ndarray flag_arr: array, data flagged as True will be taken
        :param int axis: int, the axis to apply flag
        :return obs_array_new: ObsArray, new object
        :rtype: ObsArray
        """

        axis = self.__check_axis__(axis=axis)
        if axis == self.ndim_ - 1:
            obs_array_new = self.take_by_flag_along_time(
                    flag_arr=flag_arr, **kwargs)
        elif axis == 0:
            array_map_new = self.array_map_.take_by_flag(flag_arr=flag_arr)
            obs_array_new = super(ObsArray, self).take_by_flag_along_axis(
                    flag_arr=flag_arr, axis=axis, array_map=array_map_new,
                    **kwargs)
        else:
            obs_array_new = super(ObsArray, self).take_by_flag_along_axis(
                    flag_arr=flag_arr, axis=axis, **kwargs)

        return obs_array_new

    def take_by_array_map(self, array_map_use, **kwargs):
        """
        Take the pixels contained in the input array_map_use, and sort data_
        according to the input array_map_use, return a new ObsArray object with
        the new array_map. This method can be used to take a subarray, or to
        sort the current data by using a sorted array_map

        :param ArrayMap array_map_use: ArrayMap, must be a subset of the current
            array_map_
        :return obs_array_new: ObsArray, new object taken and sorted according
            to the input array_map_use
        :rtype: ObsArray
        :raises ValueError: input array_map not a subset
        """

        array_map_use = ArrayMap(array_map_use)
        if array_map_use not in self.array_map_:
            raise ValueError("Input array map should be a subset of the " +
                             "object array_map_.")

        idxs_arr = np.asarray([
            self.array_map_.get_index_where(spat_spec=[spat, spec])[0] for
            (spat, spec) in array_map_use.array_idxs_])
        data_use = self.data_[idxs_arr] if len(idxs_arr) > 0 else \
            np.empty((0,) + self.shape_[1:], dtype=self.dtype_)

        return super(ObsArray, self).replace(
                arr_in=data_use, array_map=array_map_use, **kwargs)

    def take_where(self, spec=None, spec_ran=None, spec_list=None,
                   spat=None, spat_ran=None, spat_list=None,
                   spat_spec=None, spat_spec_list=None,
                   row=None, row_ran=None, row_list=None,
                   col=None, col_ran=None, col_list=None,
                   row_col=None, row_col_list=None,
                   logic="and", **kwargs):
        """
        take the pixels in ObeArray object that match the search condition.
        The search condition follows the same rule as in ArrayMap.flag_where()
        method. This method is equivalent to calling
        take_by_array_map(array_map_use=self.array_map_.take_where(
            **conditions), **kwargs)

        :param int spec: int, the specific spectral index to search
        :param spec_ran: list or tuple or array, will search for pixels with
            spec_ran.min() <= spectral index <= spec_ran.max()
        :type spec_ran: list or tuple or numpy.ndarray
        :param spec_list: list or tuple or array, will search for pixels at the
            spectral indices in spec_list
        :type spec_list: list or tuple or numpy.ndarray
        :param int spat: int, the specific spatial position to search
        :param spat_ran: list or tuple or array, will search for pixels
            with spat_ran.min() <= spatial position <= spat_ran.max()
        :type spat_ran: list or tuple or numpy.ndarray
        :param spat_list: list or tuple or array, will search for pixels at the
            spatial positions in spat_list
        :type spat_list: list or tuple or numpy.ndarray
        :param spat_spec: list or tuple or array, (spat, spec) of the pixel
        :type spat_spec: list or tuple or numpy.ndarray
        :param spat_spec_list: list or tuple or array, a list of (spat, spec) of
            the pixels
        :type spat_spec_list: list or tuple or numpy.ndarray
        :param int row: int, the specific MCE row to search
        :param row_ran: list or tuple or array, will search for pixels with
            row_ran.min() <= MCE row <= row_ran.max()
        :type row_ran: list or tuple or numpy.ndarray
        :param list row_list: list or tuple or array, will search for pixels at
            MCE row indices in row_list
        :type row_list: list or tuple or numpy.ndarray
        :param int col: int, the specific MCE column to search
        :param col_ran: list or tuple or array, will search for pixels with
            col_ran.min() <= MCE column <= col_ran.max()
        :type col_ran: list or tuple or numpy.ndarray
        :param col_list: list or tuple or array, will search for pixels at MCE
            column indices in col_list
        :type col_list: list or tuple or numpy.ndarray
        :param row_col: list or tuple or array, (row, col) of the pixel
        :type row_col: list or tuple or numpy.ndarray
        :param row_col_list: list or tuple or array, a list of (row, col) of the
            pixels
        :type row_col_list: list or tuple or numpy.ndarray
        :param logic: str or bool or int, if input 'and'(case insensitive) or
            '&' or True or 1, then only pixels meeting all the input selection
            conditions are flagged; if input 'or' or '|' or False or 0, then
            pixels meeting any of the selection conditions are flagged
        :type logic: str or bool or int
        :return: ObsArray, new object containing the pixels in the
            array map that match any/all of the criteria
        :rtype: ObsArray
        """

        return self.take_by_array_map(array_map_use=self.array_map_.take_where(
                spec=spec, spec_ran=spec_ran, spec_list=spec_list,
                spat=spat, spat_ran=spat_ran, spat_list=spat_list,
                spat_spec=spat_spec, spat_spec_list=spat_spec_list,
                row=row, row_ran=row_ran, row_list=row_list,
                col=col, col_ran=col_ran, col_list=col_list,
                row_col=row_col, row_col_list=row_col_list,
                logic=logic), **kwargs)

    def exclude_where(self, spec=None, spec_ran=None, spec_list=None,
                      spat=None, spat_ran=None, spat_list=None,
                      spat_spec=None, spat_spec_list=None,
                      row=None, row_ran=None, row_list=None,
                      col=None, col_ran=None, col_list=None,
                      row_col=None, row_col_list=None,
                      logic="and", **kwargs):
        """
        Return an object with the pixels in the ObeArray object that match the
        search condition excluded. The keywords are the same for
        ObsArray.take_where().

        :return: ObsArray, new object containing the pixels in the
            array map that match any of/all the criteria
        :rtype: ObsArray
        """

        return self.take_by_array_map(
                array_map_use=self.array_map_.exclude_where(
                        spec=spec, spec_ran=spec_ran, spec_list=spec_list,
                        spat=spat, spat_ran=spat_ran, spat_list=spat_list,
                        spat_spec=spat_spec, spat_spec_list=spat_spec_list,
                        row=row, row_ran=row_ran, row_list=row_list,
                        col=col, col_ran=col_ran, col_list=col_list,
                        row_col=row_col, row_col_list=row_col_list,
                        logic=logic), **kwargs)

    def chunk_split(self, *args, **kwargs):
        """
        Same as Obs.chunk_split()

        :return obs_array_chunk_list: list, containing ObsArray objects of data
            chunks split by chunk_edge_idxs
        :rtype: list
        """

        obs_array_chunk_list = super(ObsArray, self).chunk_split(
                *args, **kwargs)
        for obs_array in obs_array_chunk_list:
            obs_array.updata_array_map(self.array_map_)

        return obs_array_chunk_list

    def to_obs(self, array_type="tes", fill_value=np.nan):
        """
        Reshape the data to (spat, spec, ...) configuration and return as an
        Obs object. All the elements not in the array map will be set to
        fill_value.

        :param str or Obs or ObsArray array_type: str or Obs or ObsArray,
            the arrangement of the data_, will use [spat, spec] if array_type=
            "tes", will use
            allowed values are 'mce' and 'tes', if input is Obs object, 'mce'
            will be used, if ObsArray, 'tes' will be used
        :param float fill_value: float, value to fill in the elements not in the
            array map
        :return obs_array
        """

        new_shape = (self.array_map_.array_spat_ulim_ + 1,
                     self.array_map_.array_spec_ulim_ + 1) + self.shape_[1:]
        data_new = np.full(shape=new_shape, fill_value=fill_value,
                           dtype=self.dtype_)
        for i, (spat, spec) in enumerate(self.array_map_.array_idxs_):
            data_new[spat, spec, ...] = self.data_[i]
        obs = Obs(arr_in=data_new, chop=self.chop_, ts=self.ts_,
                  obs_info=self.obs_info_, obs_id=self.obs_id_,
                  obs_id_list=self.obs_id_list_, obs_id_arr=self.obs_id_arr_,
                  t_start_time=self.t_start_time_,
                  t_end_time=self.t_end_time_)

        return obs

    def to_mce_shape(self, fill_value=np.nan, mce_shape=(33, 24)):
        """
        Reshape the data to (row, col, ...) configuration and return as an
        Obs object. All the elements not in the array map will be set to
        fill_value.

        :param float fill_value: float, value to fill in the elements not in the
            array map
        :param tuple mce_shape: tuple, (row, col) shape of the whole MCE array
        :return obs_array
        """

        if (mce_shape[0] < self.array_map_.mce_row_ulim_ + 1) or \
                (mce_shape[1] < self.array_map_.mce_col_ulim_ + 1):
            raise ValueError("ArrayMap of MCE shape (%i, %i) " % (
                self.array_map_.mce_row_ulim_ + 1,
                self.array_map_.mce_col_ulim_ + 1) +
                             "can not be reshaped to (%i, %i)" % mce_shape[:2])
        new_shape = mce_shape[:2] + self.shape_[1:]
        data_new = np.full(shape=new_shape, fill_value=fill_value,
                           dtype=self.dtype_)
        for i, (row, col) in enumerate(self.array_map_.mce_idxs_):
            data_new[row, col, ...] = self.data_[i]
        obs = Obs(arr_in=data_new, chop=self.chop_, ts=self.ts_,
                  obs_info=self.obs_info_, obs_id=self.obs_id_,
                  obs_id_list=self.obs_id_list_, obs_id_arr=self.obs_id_arr_,
                  t_start_time=self.t_start_time_,
                  t_end_time=self.t_end_time_)

        return obs

    def to_table(self, orientation="horizontal"):
        """
        Return a table representation of ObsArray data.

        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, columns of table will be
            (spatial_position, spectral_index=0, spectral_index=1, ...),
            otherwise (spectral_index, spatial_position=0, ...)
        """

        horizontal = check_orientation(orientation=orientation)

        tb = Tb(masked=True)
        if not self.empty_flag_:
            array_bound = ((self.array_map_.array_spat_llim_,
                            self.array_map_.array_spat_ulim_),
                           (self.array_map_.array_spec_llim_,
                            self.array_map_.array_spec_ulim_))[
                          ::(1 if horizontal else -1)]
            row_num, col_num = np.diff(array_bound).flatten() + 1
            colname_main, colname_repeat = \
                ("spatial_position", "spectral_index")[::(
                    1 if horizontal else -1)]

            row_idx_arr = np.arange(array_bound[0][0], array_bound[0][1] + 1,
                                    dtype=int)
            row_idx_arr = np.repeat(row_idx_arr, self.len_)
            ts_arr = self.ts_.data_ if not self.ts_.empty_flag_ else \
                np.arange(self.len_, dtype=np.double)
            ts_arr = np.repeat([ts_arr], row_num, axis=0).flatten()
            chop_arr = self.chop_.data_ if not self.chop_.empty_flag_ else \
                np.zeros(self.len_, dtype=bool)
            chop_arr = np.repeat([chop_arr], row_num, axis=0).flatten()
            obs_id_arr = self.obs_id_arr_.data_ if not \
                self.obs_id_arr_.empty_flag_ else \
                np.full(self.len_, fill_value="", dtype="<U40")
            obs_id_arr = np.repeat([obs_id_arr], row_num, axis=0).flatten()
            tb.add_columns([Tb.Column(row_idx_arr, name=colname_main),
                            Tb.Column(ts_arr, name="time_stamp"),
                            Tb.Column(chop_arr, name="chop_phase"),
                            Tb.Column(obs_id_arr, name="obs_id")])

            for col_repeat in range(array_bound[1][0], array_bound[1][1] + 1):
                col_data = np.ma.empty((row_num, self.len_), dtype=np.double)
                col_data.mask = True
                array_map_col = self.array_map_.take_where(
                        **{("spec" if horizontal else "spat"): col_repeat})
                for row_idx in (
                        array_map_col.array_spat_ if horizontal else
                        array_map_col.array_spec_):
                    pix_idx = self.array_map_.get_index_where(
                            spat_spec=(row_idx, col_repeat)[
                                      ::(1 if horizontal else -1)])[0]
                    col_data[row_idx - array_bound[0][0]] = \
                        self.data_[pix_idx].flatten()
                    col_data.mask[row_idx - array_bound[0][0]] = False
                col_data = col_data.flatten()
                tb.add_column(Tb.MaskedColumn(
                        col_data, name="%s=%i" % (colname_repeat, col_repeat)))

        return tb


def vstack_reconcile(tbs, **kwargs):
    """
    try to append tables with astropy.table.vstack(), in the case of
    incompatible column types, will try to convert the type to a common type
    and try again

    :param tbs: list or tuple of astropy.table objects, to be appended
    :type tbs: list or tuple or Tb
    :param kwargs: keyword arguments passed to astropy.table.vstack()
    :return: stacked table
    :rtype: Tb
    :raises TypeError: can not reconcile the type
    :raises ValueError: tbs empty
    :raises RuntimeError: tbs length invalid
    """

    if isinstance(tbs, (list, tuple)):
        tbs_len = len(tbs)
        if tbs_len == 0:
            raise ValueError("tbs is empty.")
        elif tbs_len == 1:
            table_new = Tb(tbs[0], copy=False)
        elif tbs_len == 2:
            try:
                table_new = vstack(tbs, **kwargs)
            except Exception as err:
                if "columns have incompatible types" in err.args[0]:
                    colname = err.args[0].split("'")[1]
                    tb1_tmp, tb2_tmp = Tb(tbs[0], copy=True), \
                                       Tb(tbs[1], copy=True)
                    try:
                        tb1_tmp[colname] = tb1_tmp[colname].astype(
                                tb2_tmp[colname].dtype)
                    except TypeError:
                        try:
                            tb2_tmp[colname] = \
                                tb2_tmp[colname].astype(
                                        tb1_tmp[colname].dtype)
                        except TypeError as err1:
                            raise TypeError(
                                    "can not reconcile the type " +
                                    "%s and %s of the column %s." %
                                    (tb1_tmp[colname].dtype,
                                     tb2_tmp[colname].dtype, colname)) \
                                from err
                    table_new = vstack_reconcile([tb1_tmp, tb2_tmp], **kwargs)
                    del (tb1_tmp, tb2_tmp)
                else:
                    raise err
        elif tbs_len > 2:
            try:
                table_new = vstack(tbs, **kwargs)
            except Exception as err:
                if "columns have incompatible types" in err.args[0]:
                    mid_idx = int(tbs_len / 2)
                    table_new = vstack_reconcile(
                            [vstack_reconcile(tbs[:mid_idx], **kwargs),
                             vstack_reconcile(tbs[mid_idx:], **kwargs)],
                            **kwargs)
                else:
                    raise err
        else:
            raise RuntimeError("Length of tbs is invalid.")
    elif isinstance(tbs, (Tb, Row)):
        table_new = Tb(tbs, copy=False)
    else:
        raise TypeError("Invalid input type for tbs: %s." % type(tbs))

    return table_new


def check_array_type(array_type):
    """
    Check if array_type is valid input or not. Return True if is 'mce',
    False if 'tes', raise Error if invalid input

    :param array_type: str or Obs or ObsArray, allowed string values are 'mce'
        and 'tes', or you can input the object and the function will determine
        the proper type
    :type array_type: str or Obs or ObsArray
    :return: True if is mce, False if tes
    :rtype: bool
    :raises ValueError: invalid input
    """

    if isinstance(array_type, ObsArray):
        array_type = "tes"
    elif isinstance(array_type, Obs):
        array_type = "mce"

    if array_type.lower().strip()[0] == "m":
        return True
    elif array_type.lower().strip()[0] == "t":
        return False
    else:
        raise ValueError("Invalid input array_type: %s." % array_type)


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
        will use t_start_time_ in obs. Allowed input type are float as unix time
        stamps(the same format used by zeus2), string in iso or isot format and
        object
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
        t_start = Time(t_start, format="unix")
    elif isinstance(t_start, str):
        try:
            t_start = Time(t_start, format="iso")
        except ValueError:
            try:
                t_start = Time(t_start, format="isot")
            except ValueError:
                raise ValueError("String format should be ISO or ISOT.")
    t_start = Time(t_start)
    ts0 = t_start.to_value(format="unix")
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


def real_units(bias, fb, mce_col=-1, mce_bias_r=467, dewar_bias_r=120,
               shunt_r=180E-6, alt_shunt_r=140E-6, alt_col_list=(0, 3, 4),
               dewar_fb_r=5280, butterworth_constant=1218,
               rel_fb_inductance=9, max_bias_voltage=5, max_fb_voltage=0.958,
               bias_dac_bits=16, fb_dac_bits=14):
    """
    Given an array of biases and corresponding array of feedbacks (all in DAC
    units), calculate the actual current and voltage going through the TES.
    Returns: (TES voltage array, TES current array) in Volts and Amps respectively.
    The default values are taken from Carl's script

    Modified from `zeustools/iv_tools.real_units
    <https://github.com/NanoExplorer/zeustools/blob/master/zeustools/iv_tools.py>`_
    and keeps updated

    :param bias: scalar or array, tes bias value(s) in adc unit
    :type bias: int or float or numpy.ndarray
    :param fb: scalar or array, sq1 feedback value(s) in adc unit, must have the
        shape such that bias * fb * mce_col yields valid result
    :type fb: int or float or numpy.ndarray
    :param int or numpy.ndarray mce_col: int scalar or array, the MCE column
        number of the input data, because the resistor differs on a column base
        according to  zeustools.iv_tools.real_units() description; must have the
        shape such that bias * fb * mce_col yields valid result; default -1 is
        not a physical column number, but it makes sure that the default
        shunt_r is used
    :param int or float mce_bias_r: scalar, MCE bias resistance in ohm, default
        467 ohm
    :param int or float dewar_bias_r: scalar, dewar bias resistance in ohm,
        default 120 ohm to match with the bias resistance 587 in Carl's thesis
    :param int or float shunt_r: scalar, the default shunt resistance in ohm used
        for data of MCE column not in the alt_col_list, default 180 uOhm
        corresponding to actpol_R in zeustools.iv_tools.real_units()
    :param int or float alt_shunt_r: scalar, the alternative shunt resistance in
        ohm used for data of MCE column in the alt_col_list, default
        140 uOhm corresponding to cmb_R in zeustools.iv_tools.real_units()
    :param tuple or list alt_col_list: list of MCE columns using the alternative
        shunt resistor, corresponding to cmb_shunts in zeustool.real_units()
    :param int or float dewar_fb_r: scalar, dewar feedback resistance in ohm,
        default 5280 ohm
    :param int or float butterworth_constant: scalar, when running in data mode 2
        and the low pass filter is in the loop, all signals are multiplied by
        this factor
    :param int or float rel_fb_inductance: scalar, feedback inductance ratio,
        default 9 which means for a change of 1 uA in the TES, the squid will
        have to change 9 uA to keep up
    :param int or float max_bias_voltage: scalar, maximum bias voltage in V,
        default 5
    :param int or float max_fb_voltage: scalar, maximum feedback voltage in V,
        default 0.958
    :param int bias_dac_bits: int, bias DAC bit number, default 16
    :param int fb_dac_bits: int, feedback DAC bit number, default 14
    :return: tuple(tes_voltage, tes_current)
    :rtype: tuple
    """

    col = np.reshape(mce_col, newshape=(-1, 1))
    alt_col = np.reshape(alt_col_list, newshape=(1, -1))
    shunt_r_use = np.choose(np.any(col == alt_col, axis=1),
                            (shunt_r, alt_shunt_r))  # pick the shunt_r to use

    bias_raw_voltage = bias / 2 ** bias_dac_bits * max_bias_voltage * 2
    # last factor of 2 is because voltage is bipolar
    bias_current = bias_raw_voltage / (dewar_bias_r + mce_bias_r)
    fb_real_dac = fb / butterworth_constant
    fb_raw_voltage = fb_real_dac / 2 ** fb_dac_bits * max_fb_voltage * 2
    # again, last factor of 2 is because voltage is bipolar
    fb_current = fb_raw_voltage / dewar_fb_r
    tes_current = fb_current / rel_fb_inductance

    shunt_current = bias_current - tes_current

    tes_voltage = shunt_current * shunt_r_use

    return tes_voltage, tes_current
