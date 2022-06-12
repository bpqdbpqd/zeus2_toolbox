# @Date    : 2021-02-16 16:28:42
# @Credit  : Bo Peng(bp392@cornell.edu), Cody Lamarche, Christopher Rooney
# @Name    : view.py
# @Version : 2.0
"""
Visualization of data
"""

from matplotlib import cm, colors, font_manager
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from io import BytesIO

from .zeus2_io import *


class FigFlux(Figure):
    """
    Make 2-d image of Obs object, the data_ of input Obs must be either in
        (row, col) or (row, col, 1) shape
    """

    fontsize_ = 10  # type: int # fontsize in pt
    text_fontsize_ = 5  # type: int # text fontsize in pt
    x_size_, y_size_ = 0.2, 0.2  # type: float # size of x/y increment in inch
    cmap_ = plt.get_cmap("gnuplot")
    norm_ = colors.Normalize(vmin=-1e-4, vmax=1e-4)
    flag_pix_color_ = colors.to_rgba("grey")  # type: tuple[int, int, int, int]
    # flagged pixel color
    nan_pix_color_ = colors.to_rgba("white")  # type: tuple[int, int, int, int]
    # nan value pixels

    main_axes_ = None
    cb_axes_ = None
    extent_ = (-0.5, 0.5, 0.5, -0.5)  # type: tuple[float, float, float, float]
    # (left, right, bottom, top)
    orientation_ = "horizontal"  # type: str

    def __init__(self, figsize=(10, 5), dpi=100, *args, **kwargs):
        """
        Initialize a figure instance. Other input will be passed to plt.figure()
        
        :param figsize: list or tuple, (width, height), default (10, 5)
        :type figsize: list or tuple
        :param float dpi: float
        """

        font_manager._get_font.cache_clear()
        super(FigFlux, self).__init__(figsize=figsize, dpi=dpi, *args,
                                      **kwargs)

    def __init_main_axes__(self, extent=None):
        if extent is None:
            extent = get_corr_extent(self.extent_)
        else:
            self.extent_ = extent
        figsize = (self.get_figwidth(), self.get_figheight())
        dx = self.x_size_ * abs(extent[1] - extent[0]) / figsize[0]
        x = self.fontsize_ * 3.5 / 72 / figsize[0]
        dy = self.y_size_ * abs(extent[3] - extent[2]) / figsize[1]
        y = 1 - self.fontsize_ * 4.6 / 72 / figsize[1] - dy

        ax = self.add_axes((x, y, dx, dy), frameon=False, zorder=0,
                           xlim=extent[:2], ylim=extent[2:])
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis="both", direction="out",
                       bottom=True, top=True, left=True, right=True,
                       labelbottom=False, labeltop=True,
                       labelleft=True, labelright=False)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,
                      ax.xaxis.offsetText, ax.yaxis.offsetText] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize_)
        self.main_axes_ = ax
        self.__set_ticks__()

    def __init_cb_axes__(self):
        figsize = (self.get_figwidth(), self.get_figheight())
        if self.main_axes_ is None:
            x = self.fontsize_ * 3.5 / 72 / figsize[0]
            dx = 1 - x - self.fontsize_ * 0.8 / 72
            y = self.fontsize_ * 3 / 72 / figsize[1]
            dy = self.fontsize_ * 0.45 / 72 / figsize[1]
        else:
            ax = self.main_axes_
            ax_x, ax_y, ax_dx, ax_dy = ax.get_position().bounds
            x, dx = ax_x, ax_dx
            dy = max((ax_y - self.fontsize_ * 3 / 72 / figsize[1] -
                      self.fontsize_ * 0.3 / 72 / figsize[1]),
                     self.fontsize_ * 0.7 / 72 / figsize[1])
            y = max(self.fontsize_ * 3 / 72 / figsize[1],
                    (ax_y - self.fontsize_ * 0.3 / 72 / figsize[1] - dy))

        cax = self.add_axes([x, y, dx, dy], zorder=0)
        for item in ([cax.xaxis.label, cax.xaxis.offsetText] +
                     cax.get_xticklabels()):
            item.set_fontsize(self.fontsize_)
        self.cb_axes_ = cax

    def __set_ticks__(self, xticks=True, yticks=True):
        if self.main_axes_ is not None:
            ax = self.main_axes_
            xlim, ylim = self.extent_[:2], self.extent_[2:]
            lim_round = get_extent_round((*xlim, *ylim))
            xticks = np.arange(lim_round[0], lim_round[1] + 1) if xticks else []
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            yticks = np.arange(lim_round[2], lim_round[3] + 1) if yticks else []
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def __imshow__(self, arr, extent=None, **kwargs):
        """
        Plot 2-d image, if axes already exists, will use the first axes to plot
        image, and the second axes for color bar

        :param numpy.ndarray arr: array, shape is (col, row, 4)
        :param extent: extent to use in imshow(), if left None, will use extent
             in the extent_instance variable
        :type extent: list or tuple
        :raises TypeError: invalid ran type
        """

        if self.main_axes_ is None:  # get axes
            if extent is None:
                extent = get_corr_extent(self.extent_)
            self.__init_main_axes__(extent=extent)
        ax = self.main_axes_
        use_kwargs = kwargs.copy()
        default_kwargs = {"cmap": self.cmap_, "norm": self.norm_,
                          "interpolation": "none", "origin": "upper",
                          "extent": extent, "aspect": "auto"}
        for key in default_kwargs:
            if key not in use_kwargs:
                use_kwargs[key] = default_kwargs[key]
        ax.imshow(arr, **use_kwargs)
        self.__set_ticks__()

    def __plot_colorbar__(self, **kwargs):
        mappable = cm.ScalarMappable(norm=self.norm_, cmap=self.cmap_)
        mappable._A = []

        if self.cb_axes_ is None:  # get color bar axes
            self.__init_cb_axes__()
        cax = self.cb_axes_
        self.colorbar(mappable, cax=cax, orientation="horizontal", **kwargs)

    def __imshow_obs_array__(self, obs_array, mask, pix_flag_list, ran,
                             orientation, **kwargs):
        """
        Plot ObsArray
        """

        if (obs_array.ndim_ > 2) or \
                ((obs_array.ndim_ == 2) and (obs_array.shape_[1] != 1)):
            raise ValueError("Invalid ObsArray shape: %s." %
                             str(obs_array.shape_))
        array_map = obs_array.array_map_
        idx_b = (array_map.array_spat_llim_, array_map.array_spat_ulim_ + 1,
                 array_map.array_spec_llim_, array_map.array_spec_ulim_ + 1)
        # get reshaped data
        arr = obs_array.to_array_shape(fill_value=np.nan).data_
        shape = arr.shape[:2]
        arr = arr.reshape(shape)
        # get reshaped mask, and combine pix_flag_list with mask
        if mask is not None:
            mask = ObsArray(arr_in=mask, array_map=array_map). \
                to_array_shape(fill_value=False).data_.reshape(shape)
        else:
            mask = np.full(shape, fill_value=False, dtype=bool)
        if pix_flag_list is not None:
            mask_pix_list = ObsArray(arr_in=array_map.get_flag_where(
                    spat_spec_list=pix_flag_list), array_map=array_map). \
                to_array_shape(fill_value=False).data_.reshape(shape)
            mask = mask | mask_pix_list
        arr = arr[idx_b[0]:idx_b[1], idx_b[2]:idx_b[3]]
        mask = mask[idx_b[0]:idx_b[1], idx_b[2]:idx_b[3]]

        extent = get_extent(obs=obs_array, orientation=orientation,
                            extent_offset=(-0.5, -0.5, -0.5, -0.5))
        self.__imshow_arr__(arr=arr, mask=mask, pix_flag_list=None, ran=ran,
                            orientation=orientation, extent=extent,
                            **kwargs)

        # add a layer above all other layers to block non-existing pixels
        mask_blank = ObsArray(arr_in=np.full(
                obs_array.shape_[0], fill_value=False, dtype=bool),
                array_map=array_map). \
            to_array_shape(fill_value=True).data_.reshape(shape)
        mask_blank = mask_blank.reshape(shape)
        self.imshow_flag(mask=mask_blank, flag_pix_color=colors.to_rgba(
                "white", alpha=1), orientation=orientation, extent=extent,
                         zorder=9)

    def __imshow_obs__(self, obs, mask, pix_flag_list, ran, orientation,
                       extent=None, **kwargs):
        """
        Plot Obs
        """

        if (obs.ndim_ > 3) or ((obs.ndim_ == 3) and (obs.shape_[2] != 1)):
            raise ValueError("Invalid Obs shape %s" % str(obs.shape_))
        arr = obs.data_

        self.__imshow_arr__(arr=arr, mask=mask, pix_flag_list=pix_flag_list,
                            ran=ran, orientation=orientation, extent=extent,
                            **kwargs)

    def __imshow_arr__(self, arr, mask, pix_flag_list, ran, orientation,
                       extent=None, **kwargs):
        """
        Plot array
        """

        ndim, shape = arr.ndim, arr.shape
        if (ndim > 3) or ((ndim == 3) and (shape[2] != 1)):  # set arr
            raise ValueError("Invalid array shape %s" % str(shape))
        if ndim == 3:
            arr = arr.reshape(shape[:2])
        elif ndim == 1:
            arr = arr[None, ...]
        shape = arr.shape
        if orientation is None:
            orientation = self.orientation_
        if mask is not None:  # set mask
            mask = mask.reshape(shape)
        if extent is None:  # set extent
            extent = get_extent(obs=arr, orientation=orientation,
                                extent_offset=(-0.5, -0.5, -0.5, -0.5))
        if ran is None:  # set ran
            arr_mask = np.full(shape=shape, fill_value=False, dtype=bool)
            if mask is not None:  # set mask
                arr_mask = mask
            if pix_flag_list is not None:
                flag_pix_arr = np.array(pix_flag_list, dtype=int)
                if flag_pix_arr.shape == (2,):
                    flag_pix_arr = flag_pix_arr.reshape(1, 2)
                elif flag_pix_arr.shape == (0,):
                    flag_pix_arr = np.empty((0, 2), dtype=int)
                elif (flag_pix_arr.ndim != 2) or \
                        (flag_pix_arr.shape[-1] != 2):
                    raise ValueError("Invalid format for input flag_pix_arr.")
                row_idxs, col_idxs = flag_pix_arr[:, 0], flag_pix_arr[:, 1]
                for row, col in zip(row_idxs, col_idxs):
                    arr_mask[row, col] = True
            ran = np.nanmax(abs(arr[~arr_mask]))

        # transform data to RGBA
        if "cmap" in kwargs:
            self.cmap_ = plt.get_cmap(kwargs["cmap"])
        self.norm_ = colors.Normalize(vmin=-ran, vmax=ran)
        flag_nan = np.isnan(arr)
        np.putmask(arr, np.isnan(arr), 0)  # replace nan by 0
        arr_rgba = self.cmap_(self.norm_(arr))  # get rgba values
        arr_rgba[flag_nan] = np.array([self.nan_pix_color_])  # nan pixels
        # set orientation
        if not check_orientation(orientation=orientation):
            arr_rgba = np.moveaxis(arr_rgba, 0, 1)

        self.__imshow__(arr=arr_rgba, extent=extent, zorder=0, **kwargs)
        self.imshow_flag(mask=mask, pix_flag_list=pix_flag_list,
                         flag_pix_color=self.flag_pix_color_,
                         orientation=orientation, extent=extent, zorder=5)
        self.__plot_colorbar__()

    def __text_obs_array__(self, obs_array, orientation, extent=None,
                           **kwargs):
        if (obs_array.ndim_ > 2) or \
                ((obs_array.ndim_ == 2) and (obs_array.shape_[1] != 1)):
            raise ValueError("Invalid ObsArray shape: %s." %
                             str(obs_array.shape_))
        arr = obs_array.data_.flatten()
        array_map = obs_array.array_map_

        extent = get_extent(obs_array, extent_offset=(-0.5, -0.5, -0.5, -0.5))
        if self.main_axes_ is None:  # get axes
            self.__init_main_axes__(extent=extent)
        ax = self.main_axes_
        extent_round = get_extent_round(extent=extent)
        if orientation is None:
            orientation = self.orientation_

        row_arr, col_arr = array_map.array_spat_, array_map.array_spec_
        if not check_orientation(orientation=orientation):
            row_arr, col_arr = col_arr, row_arr
        for i, (row, col) in enumerate(zip(row_arr, col_arr)):
            ax.text(x=col - 0.3, y=row + 0.3, s=arr[i],
                    fontsize=self.text_fontsize_, **kwargs)

    def __text_obs__(self, obs, orientation, extent=None, **kwargs):
        if (obs.ndim_ > 3) or ((obs.ndim_ == 3) and (obs.shape_[2] != 1)):
            raise ValueError("Invalid Obs shape %s" % str(obs.shape_))
        arr = obs.data_

        self.__text_arr__(arr=arr, orientation=orientation, extent=extent,
                          **kwargs)

    def __text_arr__(self, arr, orientation, extent=None, **kwargs):
        if extent is None:
            extent = get_corr_extent(self.extent_)
        if self.main_axes_ is None:  # get axes
            self.__init_main_axes__(extent=extent)
        ax = self.main_axes_
        extent_round = get_extent_round(extent=extent)

        ndim, shape = arr.ndim, arr.shape
        if (ndim > 3) or ((ndim == 3) and (shape[2] != 1)):  # set arr
            raise ValueError("Invalid array shape %s" % str(shape))
        if ndim == 3:
            arr = arr.reshape(shape[:2])
        elif ndim == 1:
            arr = arr[None, ...]

        if orientation is None:
            orientation = self.orientation_
        if not check_orientation(orientation=orientation):
            arr = arr.transpose()
        shape = arr.shape
        for row in range(shape[0]):
            for col in range(shape[1]):
                ax.text(x=extent_round[0] + col - 0.3,
                        y=extent_round[2] + row + 0.3,
                        s=arr[row, col], fontsize=self.text_fontsize_, **kwargs)

    def savefig(self, fname, *args, compress=True, **kwargs):
        """
        instead of using the default savefig function in matplotlib, this
        function can convert the Figure object to PIL.Image.Image object,
        converting the mode from "RGB" to "P" which reduces the file size by a
        factor of three, and then save the figure to the specified filename

        :param str fname: str, filename, passed to
            matplotlib.figure.Figure.savefig()
        :param bool compress: bool flag whether to do compress size, default True
        """

        if not compress:
            return super(FigFlux, self).savefig(fname=fname, *args, **kwargs)
        else:
            save_kwarg, fmt = kwargs.copy(), None
            if "format" in save_kwarg:
                fmt = save_kwarg["format"]
                save_kwarg.pop("format")
            with BytesIO() as buffer:  # write to memory
                super(FigFlux, self).savefig(buffer, *args, **save_kwarg)
                buffer.seek(0)
                with Image.open(buffer) as im:
                    im2 = im.convert('RGB').convert(
                            'P', palette=Image.Palette.ADAPTIVE)
                    return im2.save(fp=fname, format=fmt, optimize=True,
                                    quality=50)

    def set_title(self, title, fontsize=None):
        """
        Set title of the main axes

        :param str title: str, title to use
        :param int fontsize: int, fontsize to use for title, if left None, will
            use the value of fontsize_ instance variable
        """

        if self.main_axes_ is None:
            self.__init_main_axes__()
        if fontsize is None:
            fontsize = self.fontsize_
        ax = self.main_axes_
        ax.set_title(title)
        ax.title.set_fontsize(fontsize)

    def set_labels(self, array_type="tes", orientation=None,
                   x_labelpad=None, y_labelpad=None):
        """
        Set the axis labels of the main axes based on input data array type and
        orientation

        :param str or Obs or ObsArray array_type: str or Obs or ObsArray,
            allowed values are 'mce' and 'tes', if input is Obs object, 'mce'
            will be used, if ObsArray, 'tes' will be used
        :param str orientation: str, orientation of array, allowed values are
            'horizontal' and 'vertical'; if left None, will use the default
            orientation the object is initialized with
        :param int x_labelpad: int, labelpad value to pass to set_xlabel()
        :param int y_labelpad: int, labelpad value to pass to set_ylabel()
        """

        x_label, y_label = ("column", "row") if \
            check_array_type(array_type=array_type) else \
            ("spectral index", "spatial position")
        if orientation is None:
            orientation = self.orientation_
        x_label, y_label = (x_label, y_label) if \
            check_orientation(orientation=orientation) else (y_label, x_label)

        if self.main_axes_ is None:
            self.__init_main_axes__()
        ax = self.main_axes_
        ax.set_xlabel(x_label, labelpad=x_labelpad)
        ax.set_ylabel(y_label, labelpad=y_labelpad)

    def imshow_pixel(self, obs, mask=None, pix_flag_list=None, ran=None,
                     orientation=None, extent=None, **kwargs):
        """
        Plot obs_array as 2d image, using the cmap, flag_pix_color,
        nan_pix_color in the instance variables

        :param obs: Obs or ObsArray or DataObj or array, object to plot, should
            be length 1, and shape (row, col) or (row, col, 1) if not ObsArray
        :type obs: Obs or ObsArray or DataObj or numpy.ndarray
        :param numpy.ndarray mask: array, with the same shape as
            obs_array.data_, the pixels labeled as True will be flagged.
        :param pix_flag_list: list or tuple or array of either (row, col) for
            Obs or (spat, spec) for ObsArray of the pixels to flag. Will be
            ignored if mask is used.
        :type pix_flag_list: list or tuple or numpy.ndarray
        :param float ran: float, optional, positive, +ran and -ran will be
            passed to imshow() as vmax and vmin, if left none, then the max of
            abs non-flagged data value will be used
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, will use the default orientation the
            object is initialized with
        :param tuple extent: tuple, extent to use in imshow(), will be ignored
            is ObsArray is input
        :param kwargs: passed to plt.imshow()
        :raises ValueError: empty object input
        """

        if orientation is None:
            orientation = self.orientation_
        if isinstance(obs, Obs):
            if obs.empty_flag_:
                raise ValueError("Empty Obs input.")
            if isinstance(obs, ObsArray):
                self.__imshow_obs_array__(
                        obs_array=obs, mask=mask, pix_flag_list=pix_flag_list,
                        ran=ran, orientation=orientation, **kwargs)
                self.set_labels(array_type="tes", orientation=orientation)
            else:
                self.__imshow_obs__(
                        obs=obs, mask=mask, pix_flag_list=pix_flag_list,
                        ran=ran, orientation=orientation,
                        extent=extent, **kwargs)
                self.set_labels(array_type="mce", orientation=orientation)
        elif isinstance(obs, (DataObj, np.ndarray, list, tuple)):
            arr = DataObj(arr_in=obs).data_
            self.__imshow_arr__(
                    arr=arr, mask=mask, pix_flag_list=pix_flag_list, ran=ran,
                    orientation=orientation, extent=extent, **kwargs)
            self.set_labels(array_type="mce", orientation=orientation)
        else:
            raise TypeError("Invalid input obs_array type.")

    def imshow_text(self, obs, orientation=None, extent=None,
                    text_fontsize=None, **kwargs):
        """
        Put text in obs_array on the figure, using the fontsize in
        text_fontsize_ instance variable

        :param obs: Obs or ObsArray or DataObj or array, object to plot, should
            be shape (row, col) or (row, col, 1) if it is not ObsArray,
            otherwise will be shaped according to array map
        :type obs: Obs or ObsArray or DataObj or numpy.ndarray
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, will use the default orientation the
            object is initialized with
        :param tuple extent: tuple, extent to use in imshow(), will be ignored
            is ObsArray is input
        :param int text_fontsize: int, font size of text in point, if left None,
            will use the value in self.text_fontsize_, default 5
        :param kwargs: passed to plt.text()
        :raises ValueError: empty object input
        """

        if orientation is None:
            orientation = self.orientation_
        if text_fontsize is not None:
            self.text_fontsize_ = text_fontsize
        if isinstance(obs, Obs):
            if obs.empty_flag_:
                raise ValueError("Empty Obs input.")
            if isinstance(obs, ObsArray):
                self.__text_obs_array__(obs_array=obs, orientation=orientation,
                                        **kwargs)
                self.set_labels(array_type="tes", orientation=orientation)
            else:
                self.__text_obs__(obs=obs, orientation=orientation,
                                  extent=extent, **kwargs)
                self.set_labels(array_type="mce", orientation=orientation)
        elif isinstance(obs, (DataObj, np.ndarray, list, tuple)):
            arr = DataObj(arr_in=obs).data_
            self.__text_arr__(arr=arr, orientation=orientation, extent=extent,
                              **kwargs)
            self.set_labels(array_type="mce", orientation=orientation)
        else:
            raise TypeError("Invalid input obs_array type.")

    def imshow_flag(self, mask=None, pix_flag_list=None, flag_pix_color="grey",
                    orientation=None, extent=None, zorder=5):
        """
        Plot a layer of flagged pixel above other layers

        :param numpy.ndarray mask: array, 2-d, the first axis is row(or spat),
            the second is col(or spec)
        :param pix_flag_list: list or tuple or array of either (row, col) for
            Obs or (spat, spec) for ObsArray of the pixels to flag. Will be
            combined with mask is used if mask is used.
        :type pix_flag_list: list or tuple or numpy.ndarray
        :param flag_pix_color: str or list or tuple, name of color or RGB or RGBA
            tuple, will be passed to colors.to_rgba(), color of flagged pixel
        :type flag_pix_color: str or list or tuple
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, will use the default orientation the
            object is initialized with
        :param tuple extent: tuple, extent to use in imshow(), will be ignored
            if ObsArray is input. if left None, will use the value of extent_
        :param int zorder: int, zorder number of this layer, values above 0
            means it will be above the imshow() layer
        :raises ValueError: invalid mask shape
        """

        if extent is None:
            extent = get_corr_extent(self.extent_)
        if flag_pix_color is None:
            flag_pix_color = self.flag_pix_color_
        else:
            flag_pix_color = colors.to_rgba(flag_pix_color)
        if orientation is None:
            orientation = self.orientation_
        extent_round = get_extent_round(extent)
        shape = (extent_round[3] - extent_round[2] + 1,
                 extent_round[1] - extent_round[0] + 1, 4)
        flag_mask = np.zeros(shape)  # an all white transparent rgba array

        row_idxs, col_idxs = [], []
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("mask should be 2-d array.")
            row_idxs_mask, col_idxs_mask = np.nonzero(mask)
            row_idxs_mask = row_idxs_mask.tolist()
            col_idxs_mask = col_idxs_mask.tolist()
            if check_orientation(orientation=orientation):
                col_idxs += col_idxs_mask
                row_idxs += row_idxs_mask
            else:
                col_idxs += row_idxs_mask
                row_idxs += col_idxs_mask

        if pix_flag_list is not None:
            flag_pix_arr = np.array(pix_flag_list, dtype=int)
            if flag_pix_arr.shape == (2,):
                flag_pix_arr = flag_pix_arr.reshape(1, 2)
            elif flag_pix_arr.shape == (0,):
                flag_pix_arr = np.empty((0, 2), dtype=int)
            elif (flag_pix_arr.ndim != 2) or \
                    (flag_pix_arr.shape[-1] != 2):
                raise ValueError("Invalid format for input flag_pix_arr.")
            row_idxs_list = flag_pix_arr[:, 0].tolist()
            col_idxs_list = flag_pix_arr[:, 1].tolist()
            if check_orientation(orientation=orientation):
                row_idxs += row_idxs_list
                col_idxs += col_idxs_list
            else:
                row_idxs += col_idxs_list
                col_idxs += row_idxs_list

        for row, col in zip(row_idxs, col_idxs):
            if (extent_round[2] <= row <= extent_round[3]) and \
                    (extent_round[0] <= col <= extent_round[1]):
                flag_mask[row - extent_round[2], col - extent_round[0]] = \
                    flag_pix_color

        self.__imshow__(arr=flag_mask, extent=extent, zorder=zorder)

    @classmethod
    def init_by_extent(cls, extent=None, orientation=None, dpi=100,
                       fontsize=None, x_size=None, y_size=None, **kwargs):
        """
        Initialize a figure with the figure size that fits the extent of the
        image, extent is tuple of (left, right, bottom, top) of the axes, along
        with the main axes

        :param tuple extent: tuple of extent as used in imshow, if left None,
            class default (-0.5, 0.5, 0.5, -0.5) will be used
        :param str orientation: str, set the object default orientation for
            plotting data, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, default value 'horizontal' will be used
        :param int dpi: int, dpi, default 100
        :param int fontsize: int, font size in point, default 10
        :param float x_size: float, size of increment in extent of x-axis in
            inch, default 0.2
        :param float y_size: float, default 0.2
        :param kwargs: passed to plt.figure()
        :return: a new PlotIm or subclass object
        :rtype: PlotIm
        """

        if extent is None:
            extent = cls.extent_
        if fontsize is None:
            fontsize = cls.fontsize_
        if x_size is None:
            x_size = cls.x_size_
        if y_size is None:
            y_size = cls.y_size_
        if orientation is None:
            orientation = cls.orientation_
        width = fontsize * 3.5 / 72 + abs(extent[1] - extent[0]) * x_size + \
                fontsize * 0.8 / 72
        height = fontsize * 4.6 / 72 + abs(extent[3] - extent[2]) * y_size + \
                 fontsize * 4 / 72
        fig = plt.figure(figsize=(width, height), dpi=dpi, **kwargs,
                         FigureClass=cls)
        fig.orientation_ = "horizontal" if \
            check_orientation(orientation=orientation) else "vertical"
        fig.fontsize_ = fontsize
        fig.x_size_, fig.y_size_ = x_size, y_size
        fig.__init_main_axes__(extent=extent)

        return fig

    @classmethod
    def plot_flux(cls, obs_array, mask=None, pix_flag_list=None, ran=None,
                  orientation=None, extent=None, cmap="coolwarm",
                  flag_pix_color="grey", nan_pix_color="white",
                  dpi=100, fontsize=None, x_size=None, y_size=None):
        """
        Plot obs_array, will initialize FigFlux with figsize that fits with
        the input data shape, and then call imshow() to plot

        :param ObsArray obs_array: ObsArray, object to plot, should be length 1
        :param numpy.ndarray mask: array, with the same shape as
            obs_array.data_, the pixels labeled as True will be flagged.
        :param pix_flag_list: list or tuple or array of either (row, col) for
            Obs or (spat, spec) for ObsArray of the pixels to flag. Will be
            combined with mask is used if mask is used.
        :type pix_flag_list: list or tuple or numpy.ndarray
        :param float ran: float, optional, positive, +ran and -ran will be
            passed to imshow() as vmax and vmin, if left none, then the max of
            abs non-flagged data value will be used
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, class default 'horizontal' will be used
        :param tuple extent: tuple, extent to use in imshow(), will be ignored
            if ObsArray is input. If left none, will call get_extent()
        :param str cmap: str, name of colormap to used, passed to plt.get_cmap()
        :param flag_pix_color: str or list or tuple, name of color or RGB or
            RGBA tuple, will be passed to colors.to_rgba(), color of the flagged
            pixel
        :type flag_pix_color: str or list or tuple
        :param nan_pix_color: str or list or tuple, name of color or RGB or
            RGBA tuple, will be passed to colors.to_rgba(), color of pixels with
            nan value, and the pixels not existing in a complete square array,
            priority lower than flag_pix_color
        :type nan_pix_color: str or list or tuple
        :param int dpi: int, dpi of the figure
        :param int fontsize: int, font size of axes label and title in point,
            default 10
        :param float x_size: float, size of increment in x-axis in inch, default
            0.2
        :param float y_size: float, size of increment in y-axis in inch, default
            0.2
        :return fig: FigFlux, object containing the plot
        :rtype: FigFlux
        :raises TypeError: invalid obs_array type
        :raises ValueError: empty input
        """

        if orientation is None:
            orientation = cls.orientation_
        if extent is None:
            if isinstance(obs_array, (Obs, np.ndarray, tuple, list)):
                extent = get_extent(obs=obs_array, orientation=orientation,
                                    extent_offset=(-0.5, -0.5, -0.5, -0.5))
            else:
                raise TypeError("Invalid input type.")
        fig = cls.init_by_extent(extent=extent, orientation=orientation, dpi=dpi,
                                 fontsize=fontsize, x_size=x_size, y_size=y_size)
        fig.cmap_ = plt.get_cmap(cmap)
        fig.flag_pix_color_ = colors.to_rgba(flag_pix_color)
        fig.nan_pix_color_ = colors.to_rgba(nan_pix_color)

        fig.imshow_pixel(obs=obs_array, mask=mask, pix_flag_list=pix_flag_list,
                         ran=ran, orientation=orientation, extent=extent)
        fig.imshow_flag(mask=mask, pix_flag_list=pix_flag_list)

        return fig

    @classmethod
    def write_text(cls, obs_array, orientation=None,
                   extent=None, text_fontsize=None, dpi=100, fontsize=None,
                   x_size=None, y_size=None):
        """
        Plot obs_array, will initialize FigFlux with figsize that fits with the
        data shape, and then call imshow() to plot

        :param obs_array: Obs or ObsArray or DataObj or array, object to plot,
            should be shape (row, col) or (row, col, 1) if it is not ObsArray,
            otherwise will be shaped according to array map
        :type obs_array: Obs or ObsArray or DataObj or numpy.ndarray
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if the former, layout of image (y, x) will be (row, col)
            for Obs and (spat, spec) for ObsArray, otherwise (col, row) and
            (spec, spat); if left None, class default 'horizontal' will be used
        :param tuple extent: tuple, extent to use in imshow(), will be ignored
            if ObsArray is input. If left none, will call get_extent()
        :param int text_fontsize: int, text font size in pt, default 5
        :param int dpi: int, dpi of the figure
        :param int fontsize: int, font size of axes label and title in point
        :param float x_size: float, size of increment in x-axis in inch
        :param float y_size: float, size of increment in y-axis in inch
        :return fig: FigFlux, object containing the plot
        :rtype: FigFlux
        :raises ValueError: empty input
        """

        if orientation is None:
            orientation = cls.orientation_
        if (extent is None) or isinstance(obs_array, ObsArray):
            extent = get_extent(obs=obs_array, orientation=orientation,
                                extent_offset=(-0.5, -0.5, -0.5, -0.5))
        fig = cls.init_by_extent(extent=extent, dpi=dpi, fontsize=fontsize,
                                 x_size=x_size, y_size=y_size)

        fig.imshow_text(obs=obs_array, orientation=orientation,
                        extent=extent, text_fontsize=text_fontsize)

        return fig


class FigArray(FigFlux):
    """
    Plot time stream of each pixel on either observation or MCE array layout
    """

    fontsize_ = 20  # type: int # fontsize of outer axes
    text_fontsize_ = 10  # type: int
    axs_fontsize_ = 6  # type: int # fontsize of axes for each pixel
    x_size_, y_size_ = 1.8, 1.5  # type: float # size axes for each pixel in inch
    axs_list_ = None  # type: list
    twin_axs_list_ = None  # type: list
    array_map_ = None  # type: ArrayMap

    def __init_axs_list__(self, array_map=None, orientation=None):
        self.array_map_ = array_map
        if orientation is None:
            orientation = self.orientation_
        else:
            orientation = "horizontal" if \
                check_orientation(orientation=orientation) else "vertical"
            self.orientation_ = orientation
        if self.main_axes_ is None:
            extent = get_extent(obs=array_map, orientation=orientation,
                                extent_offset=(
                                    - 0.5 - 1.5 * self.axs_fontsize_ / 72 / self.x_size_,
                                    - 0.5 + 1.5 * self.axs_fontsize_ / 72 / self.x_size_,
                                    - 0.5 + 2 * self.axs_fontsize_ / 72 / self.y_size_,
                                    - 0.5 - 2 * self.axs_fontsize_ / 72 / self.y_size_))
            self.__init_main_axes__(extent=extent)
        ax = self.main_axes_
        extent_round = (array_map.array_spec_llim_, array_map.array_spec_ulim_,
                        array_map.array_spat_llim_, array_map.array_spat_ulim_,)
        row_idxs, col_idxs = array_map.array_spat_, array_map.array_spec_
        if not check_orientation(orientation=orientation):
            extent_round = (*extent_round[2:], *extent_round[:2])
            row_idxs, col_idxs = col_idxs, row_idxs

        ax_x, ax_y = ax.get_position().bounds[:2]
        figsize = (self.get_figwidth(), self.get_figheight())
        dx = (self.x_size_ - 3 * self.axs_fontsize_ / 72) / figsize[0]
        x = ax_x + 3 * self.axs_fontsize_ / 72 / figsize[0]
        dy = (self.y_size_ - 4 * self.axs_fontsize_ / 72) / figsize[1]
        y = 1 - self.fontsize_ * 4.6 / 72 / figsize[1] - \
            4 * self.axs_fontsize_ / 72 / figsize[1] - dy

        axs_list = []
        for row, col in zip(row_idxs, col_idxs):
            axp = self.add_axes(
                    (x + (col - extent_round[0]) * self.x_size_ / figsize[0],
                     y - (row - extent_round[2]) * self.y_size_ / figsize[1],
                     dx, dy), zorder=10)
            axp.patch.set_alpha(0.5)
            axp.tick_params(axis="both", direction="in",
                            bottom=True, top=True, left=True, right=True)
            axp.tick_params(axis="y", labelrotation=90)
            for item in ([axp.xaxis.label, axp.yaxis.label,
                          axp.xaxis.offsetText, axp.yaxis.offsetText] +
                         axp.get_xticklabels() + axp.get_yticklabels()):
                item.set_fontsize(self.axs_fontsize_)
            axs_list.append(axp)
        self.axs_list_ = axs_list

    def __init_twin_axs_list__(self):
        if self.axs_list_ is None:
            raise ValueError("axs_list_ not initialized.")

        twin_axs_list = []
        for ax in self.axs_list_:
            twin_ax = ax.twinx()
            twin_ax.tick_params(axis="both", direction="in",
                                bottom=True, top=True, left=True, right=True)
            twin_ax.tick_params(axis="y", labelrotation=90)
            for item in ([twin_ax.yaxis.label, twin_ax.yaxis.offsetText] +
                         twin_ax.get_yticklabels()):
                item.set_fontsize(self.axs_fontsize_)
            twin_axs_list.append(twin_ax)
        self.twin_axs_list_ = twin_axs_list

    def set_xlim(self, xlim=None):
        """
        Set xlim on pixel base if xlim is Obs/ObsArray object, or set for all
        the pixels if xlim is a tuple. if xlim is left None, then xlim will take
        the value of the minimum and maximum of xlim of all axes, to make all
        axes to share the same xlim.

        :param xlim: xlim to set for individual (Obs/ObsArray input) or all
            (tuple input) pixel axes. If left None, will pick the xlim suitable
            xlim for all pixel axes
        :type xlim: tuple or list or Obs or ObsArray or None
        :raises ValueError: axes list not initialized
        :raises TypeError: invalid xlim type
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling xlim.")
        axs_use = self.axs_list_

        if xlim is None:
            xlim_list = []
            for ax in axs_use:
                if len(ax.lines) + len(ax.collections) > 0:
                    xlim_list.append(ax.get_xlim())
            if len(xlim_list) == 0:
                xlim_list = [(0, 1)]
            xlim = (np.min(xlim_list), np.max(xlim_list))
        if isinstance(xlim, (tuple, list, np.ndarray)):
            for ax in axs_use:
                ax.set_xlim(xlim)
        elif isinstance(xlim, Obs):
            xlim = ObsArray(xlim)
            array_map = xlim.array_map_
            for (spat, spec), lim in zip(array_map.array_idxs_, xlim.data_):
                ax_idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    axp.set_xlim(lim)
        else:
            raise TypeError("Invalid input type for xlim.")

    def set_ylim(self, ylim=None, twin_axes=False):
        """
        Set ylim on pixel base if ylim is Obs/ObsArray object, or set for all
        the pixels if ylim is a tuple. if ylim is left None, then ylim will take
        the value of the minimum and maximum of ylim of all axes, to make all
        axes to share the same ylim.

        :param ylim: ylim to set for individual (Obs/ObsArray input) or all
            (tuple input) pixel axes. If left None, will pick the ylim suitable
            ylim for all pixel axes
        :type ylim: tuple or list or Obs or ObsArray or None
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :raises ValueError: axes list not initialized
        :raises TypeError: invalid ylim type
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling ylim.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        if ylim is None:
            ylim_list = []
            for ax in axs_use:
                if len(ax.lines) + len(ax.collections) > 0:
                    ylim_list.append(ax.get_ylim())
            if len(ylim_list) == 0:
                ylim_list = [(0, 1)]
            ylim = (np.min(ylim_list), np.max(ylim_list))
        if isinstance(ylim, (tuple, list, np.ndarray)):
            for ax in axs_use:
                ax.set_ylim(ylim)
        elif isinstance(ylim, Obs):
            ylim = ObsArray(ylim)
            array_map = ylim.array_map_
            for (spat, spec), lim in zip(array_map.array_idxs_, ylim.data_):
                ax_idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    axp.set_ylim(lim)
        else:
            raise TypeError("Invalid input type for ylim.")

    def set_xlabel(self, xlabel, **kwargs):
        """
        set xlabel

        :param str xlabel: str
        :param kwargs: passed to axes.set_xlabel()
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling xlabel.")
        axs_use = self.axs_list_
        func = np.max
        edge_spat_spec_list = []
        if check_orientation(self.orientation_):
            for spec in range(self.array_map_.array_spec_llim_,
                              self.array_map_.array_spec_ulim_ + 1):
                array_map_use = self.array_map_.take_where(spec=spec)
                if not array_map_use.empty_flag_:
                    spat = func(array_map_use.array_spat_)
                    edge_spat_spec_list.append((spat, spec))
        else:
            for spat in range(self.array_map_.array_spat_llim_,
                              self.array_map_.array_spat_ulim_ + 1):
                array_map_use = self.array_map_.take_where(spat=spat)
                if not array_map_use.empty_flag_:
                    spec = func(array_map_use.array_spec_)
                    edge_spat_spec_list.append((spat, spec))

        for spat_spec in edge_spat_spec_list:
            idx = self.array_map_.get_index_where(spat_spec=spat_spec)
            ax = axs_use[idx[0]]
            ax.set_xlabel(xlabel, **kwargs)

    def set_ylabel(self, ylabel, twin_axes=False, **kwargs):
        """
        set ylabel

        :param str ylabel: str
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :param kwargs: passed to axes.set_xlabel()
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling ylabel.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
            func = np.max
        else:
            axs_use = self.axs_list_
            func = np.min

        edge_spat_spec_list = []
        if not check_orientation(self.orientation_):
            for spec in range(self.array_map_.array_spec_llim_,
                              self.array_map_.array_spec_ulim_ + 1):
                array_map_use = self.array_map_.take_where(spec=spec)
                if (not array_map_use.empty_flag_) and len(array_map_use) > 0:
                    spat = func(array_map_use.array_spat_)
                    edge_spat_spec_list.append((spat, spec))
        else:
            for spat in range(self.array_map_.array_spat_llim_,
                              self.array_map_.array_spat_ulim_ + 1):
                array_map_use = self.array_map_.take_where(spat=spat)
                if (not array_map_use.empty_flag_) and len(array_map_use) > 0:
                    spec = func(array_map_use.array_spec_)
                    edge_spat_spec_list.append((spat, spec))

        for spat_spec in edge_spat_spec_list:
            idx = self.array_map_.get_index_where(spat_spec=spat_spec)
            ax = axs_use[idx[0]]
            ax.set_ylabel(ylabel, **kwargs)

    def set_xscale(self, xscale="linear", **kwargs):
        """
        set x-axis scale using axes.set_xscale()

        :param str xscale: value{"linear", "log", "symlog", "logit", ...},
            passed to axes.set_xscale()
        :param kwargs: passed to axes.set_xscale()
        """
        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling ylabel.")
        axs_use = self.axs_list_

        for ax in axs_use:
            ax.set_xscale(xscale, **kwargs)

    def set_yscale(self, yscale="linear", twin_axes=False, **kwargs):
        """
        set y-axis scale using axes.set_yscale()

        :param str yscale: value{"linear", "log", "symlog", "logit", ...},
            passed to axes.set_yscale()
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :param kwargs: passed to axes.set_yscale()
        """
        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling ylabel.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for ax in axs_use:
            ax.set_yscale(yscale, **kwargs)

    def legend(self, twin_axes=False, *args, **kwargs):
        """
        Show legend on the uppermost axes of the leftmost column, with arg and
        kwargs passed to axes.legend()
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling legend.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        func = np.min

        if check_orientation(self.orientation_):
            spec = self.array_map_.array_spec_llim_
            array_map_use = self.array_map_.take_where(spec=spec)
            spat = func(array_map_use.array_spat_)
        else:
            spat = self.array_map_.array_spat_llim_
            array_map_use = self.array_map_.take_where(spat=spat)
            spec = func(array_map_use.array_spec_)

        idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
        ax = axs_use[idx[0]]
        if "fontsize" not in kwargs:
            kwargs["fontsize"] = self.axs_fontsize_
        ax.legend(*args, **kwargs)

    def scatter(self, obs_array, s=0.2, c=None, marker=".", twin_axes=False,
                **kwargs):
        """
        Scatter plot in pixel axes. If no color is specified and obs_array has
        chop_ initialized, then will plot off chop as blue and on chop red,
        otherwise default color is black. If ts_ exists, it will be used as
        x-axis, otherwise will auto generate x-axis.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            single time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param float s: float, size of marker
        :param c: str or rgba tuple or array
        :type c: str or tuple or numpy.ndarray
        :param str marker: str, marker
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keywords to pass to axes.scatter()
        """

        obs_array = ObsArray(arr_in=obs_array)
        array_map = obs_array.array_map_
        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map,
                                   orientation=self.orientation_)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        ts = obs_array.ts_
        x = ts.data_ if not ts.empty_flag_ else np.arange(obs_array.len_)

        c_flag = ("color" in kwargs)
        if (c is None) and not c_flag:
            c = get_chop_color(chop=obs_array.chop_)

        for (spat, spec), data in zip(array_map.array_idxs_, obs_array.data_):
            if np.count_nonzero(np.isfinite(data)) > 0:
                ax_idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    axp.scatter(x=x, y=data, s=s, c=c, marker=marker, **kwargs)

    def plot(self, obs_array, *args, twin_axes=False, **kwargs):
        """
        Plot in pixel axes. If ts_ exists, it will be used as x-axis, otherwise
        will auto generate x-axis. *args and **kwargs are parameters passed to
        plt.plot() right after x and y. This method has the most flexibility.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            single time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keywords to pass to axes.plot()
        """

        obs_array = ObsArray(arr_in=obs_array)
        array_map = obs_array.array_map_
        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map,
                                   orientation=self.orientation_)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        ts = obs_array.ts_
        x = ts.data_ if not ts.empty_flag_ else np.arange(obs_array.len_)

        for (spat, spec), data in zip(array_map.array_idxs_, obs_array.data_):
            if np.count_nonzero(np.isfinite(data)) > 0:
                ax_idx = self.array_map_.get_index_where(spat=spat, spec=spec)
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    axp.plot(x, data, *args, **kwargs)

    def errorbar(self, obs_array, yerr=None, xerr=None, fmt='.', color=None,
                 twin_axes=False, **kwargs):
        """
        Plot error bar in pixel axes. If no color is specified and obs_array has
        chop_ initialized, then will plot off chop as blue and on chop red,
        otherwise default color is black. If ts_ exists, it will be used as
        x-axis, otherwise will auto generate x-axis.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            single time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param yerr: Obs or ObsArray or DataObj or array, must have the same
            shape as obs_array
        :type yerr: Obs or ObsArray or DataObj or numpy.ndarray
        :param xerr: Obs or ObsArray or DataObj or array, must have the same
            shape as obs_array
        :type yerr: Obs or ObsArray or DataObj or numpy.ndarray
        :param str fmt: str, by default '.'
        :param color: str or tuple or array of rgba
        :type color: str or tuple or numpy.ndarray
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keywords to pass to plt.errorbar()
        """

        obs_array = ObsArray(arr_in=obs_array)
        array_map = obs_array.array_map_
        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map,
                                   orientation=self.orientation_)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        ts = obs_array.ts_
        x = ts.data_ if not ts.empty_flag_ else np.arange(obs_array.len_)
        xerr_data = [None] * obs_array.shape_[0] if xerr is None else ObsArray(
                arr_in=xerr).take_by_array_map(array_map_use=array_map).data_
        yerr_data = [None] * obs_array.shape_[0] if yerr is None else ObsArray(
                arr_in=yerr).take_by_array_map(array_map_use=array_map).data_

        c_flag = ("c" in kwargs) or ("ecolor" in kwargs) or \
                 ("markeredgecolor" in kwargs) or ("mec" in kwargs) or \
                 ("markerfacecolor" in kwargs) or ("mfc" in kwargs)
        if (color is None) and not c_flag:
            color = get_chop_color(chop=obs_array.chop_)

        for (spat, spec), data, yerr, xerr in zip(
                array_map.array_idxs_, obs_array.data_, yerr_data, xerr_data):
            if np.count_nonzero(np.isfinite(data)) > 0:
                ax_idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    if not c_flag:
                        axp.errorbar(x=x, y=data, yerr=yerr, xerr=xerr, fmt=fmt,
                                     ecolor=color, **kwargs)
                    else:
                        axp.errorbar(x=x, y=data, yerr=yerr, xerr=xerr, fmt=fmt,
                                     **kwargs)

    def imshow(self, obs_array, twin_axes=False, **kwargs):
        """
        Plot image on every pixel axes.

        :param obs_array: Obs or ObsArray, the data to plot is in the last 2
            axis, the last axis will be plotted along x-axis, and the second
            last on is the y-axis.
        :param bool twin_axes: bool flag, whether to use the secondary y axes
        :param kwargs: passed to axes.imshow()
        """

        obs_array = ObsArray(arr_in=obs_array)
        array_map = obs_array.array_map_
        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map,
                                   orientation=self.orientation_)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for (spat, spec), data in zip(array_map.array_idxs_, obs_array.data_):
            if np.count_nonzero(np.isfinite(data)) > 0:
                ax_idx = self.array_map_.get_index_where(spat_spec=(spat, spec))
                if len(ax_idx) > 0:
                    axp = axs_use[ax_idx[0]]
                    axp.imshow(data, **kwargs)

    def psd(self, obs_array, freq_ran=(0.5, 6), scale="linear", xscale="linear",
            twin_axes=False, **kwargs):
        """
        Make power spectral density spectrum for obs_array. Call fft_obs() on
        obs_array, then plot the psd of the fft result for only the specified
        freq_ran

        :param obs_array: Obs or ObsArray, object to calculate power spectral
            density and to plot, must have ts_ initialized
        :type obs_array: Obs or ObsArray
        :param freq_ran: tuple or list or array, the shown spectrum will be cut
            by freq_ran[0] <= freq <= freq_ran[1]
        :type freq_ran: tuple or list or numpy.ndarray
        :param str scale: str flag of the scale, value{"linear", "log", "symlog",
            "logit", ...} will be passed to set_yscale() method, value{"db"}
            will convert power to db as 10*log10(amp**2) and then plot in linear
            scale
        :param str xscale: str, passed to set_xscale() method setting the scale
            of the plotted x-axis
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keyword arguments passed to plot()
        :raises ValueError: invalid scale value
        """

        obs_array = ObsArray(arr_in=obs_array)
        obs_fft = fft_obs(obs_array)
        obs_fft = obs_fft.replace(
                arr_in=np.fft.fftshift(obs_fft.data_, axes=-1),
                ts=np.fft.fftshift(obs_fft.ts_.data_))
        obs_fft = obs_fft.take_when(t_ran=freq_ran)

        obs_spec = abs(obs_fft) ** 2
        if scale.strip().lower()[:2] == "db":
            obs_spec = 10 * obs_spec.log10()
            scale = "linear"

        self.plot(obs_spec, twin_axes=twin_axes, **kwargs)
        self.set_xscale(xscale)
        self.set_yscale(scale)

    def specgram(self, obs_array, nfft=5., noverlap=4., freq_ran=(0.5, 4.5),
                 scale="linear", twin_axes=False, **kwargs):
        """
        Plot spectrogram for obs_array. Call nfft_obs() on obs_array, then plot
        spectrogram for the nfft result on the specified freq_ran using imshow()

        :param obs_array: Obs or ObsArray, object to calculate power spectral
            density and to plot, must have ts_ initialized
        :type obs_array: Obs or ObsArray
        :param nfft: number of data points in each block to do fft for int
            input, or seconds of data for each block for float input
        :type nfft: int or float
        :param noverlap: number of data points that overlap between blocks, or
            seconds of overlapping data points between blacks. If the value of
            noverlap >= nfft, then nfft number of data point - 1 will be used
        :type noverlap: int or float
        :param freq_ran: tuple or list or array, the shown spectrum will be cut
            by freq_ran[0] <= freq <= freq_ran[1]
        :type freq_ran: tuple or list or numpy.ndarray
        :param str scale: str flag of the scale, only "linear", "log" and "dB"
            are accepted
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keyword arguments passed to imshow()
        :raises ValueError: invalid scale value
        """

        obs_array = ObsArray(arr_in=obs_array)
        obs_nfft, freq_ts = nfft_obs(obs_array, nfft=nfft, noverlap=noverlap)
        obs_nfft = obs_nfft.replace(
                arr_in=np.fft.fftshift(obs_nfft.data_, axes=-2))
        freq_ts = freq_ts.replace(arr_in=np.fft.fftshift(freq_ts.data_))
        freq_flag = (freq_ts.data_ >= freq_ran[0]) & \
                    (freq_ts.data_ <= freq_ran[1])
        obs_nfft = obs_nfft.take_by_flag_along_axis(flag_arr=freq_flag, axis=-2)
        freq_ts = freq_ts.take_by_flag_along_axis(flag_arr=freq_flag)

        obs_specgram = abs(obs_nfft) ** 2
        if scale.strip().lower()[:2] in ["db", "lo", "ln"]:
            obs_specgram = 10 * obs_specgram.log10()
        elif scale.strip().lower()[:2] == "li":
            pass
        else:
            raise ValueError("Invalid input value for scale.")
        if "extent" not in kwargs:
            extent = (obs_specgram.ts_.t_start_, obs_specgram.ts_.t_end_,
                      freq_ts.t_start_, freq_ts.t_end_)
            if ("origin" not in kwargs) or \
                    (kwargs["origin"].strip().lower()[0] == "l"):
                kwargs["extent"] = extent
                kwargs["origin"] = "lower"
            else:
                kwargs["extent"] = (*extent[:2], *extent[-2:][::-1])
        if "aspect" not in kwargs:
            kwargs["aspect"] = "auto"

        self.imshow(obs_specgram, twin_axes=twin_axes, **kwargs)

    def plot_all_pixels(self, *args, twin_axes=False, **kwargs):
        """
        plot the same data in all the pixel axes. All the args and kwargs inputs
        will be passed to axes.plot in every pixel axes.

        :param bool twin_axes: bool flag, whether to plot using the secondary
            y-axis
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize pixel axes using ArrayMap " +
                             "before calling plot_all_pixels.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for ax in axs_use:
            ax.plot(*args, **kwargs)

    @classmethod
    def init_by_array_map(cls, array_map, orientation=None, dpi=150,
                          fontsize=None, axs_fontsize=None,
                          x_size=None, y_size=None, **kwargs):
        """
        Initialize a figure with the size fit for the input array_map, with axes
        corresponding to the pixels in array_map

        :param ArrayMap array_map: ArrayMap of the array layout, the spat and
            spec information will be used
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if 'horizontal', row will be spatial positional and
            column will be spectral position; if left None, class default
            'horizontal' will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 6
        :param float x_size: float, size of pixel in figure in inch, default 1.8
        :param float y_size: float, default 1.5
        :param kwargs: keywords passed to plt.figure()
        :return: FigArray object
        :rtype: FigArray
        """

        array_map = ArrayMap(array_map) if not isinstance(array_map, Obs) else \
            ObsArray(arr_in=array_map).array_map_
        if fontsize is None:
            fontsize = cls.fontsize_
        if axs_fontsize is None:
            axs_fontsize = cls.axs_fontsize_
        if x_size is None:
            x_size = cls.x_size_
        if y_size is None:
            y_size = cls.y_size_
        if orientation is None:
            orientation = cls.orientation_

        extent = get_extent(
                obs=array_map, orientation=orientation, extent_offset=(
                    - 0.5 - 1.5 * axs_fontsize / 72 / x_size,
                    - 0.5 + 1.5 * axs_fontsize / 72 / x_size,
                    - 0.5 + 2 * axs_fontsize / 72 / y_size,
                    - 0.5 - 2 * axs_fontsize / 72 / y_size))
        fig = cls.init_by_extent(extent=extent, orientation=orientation, dpi=dpi,
                                 fontsize=fontsize, x_size=x_size, y_size=y_size,
                                 **kwargs)
        fig.axs_fontsize_ = axs_fontsize
        fig.__init_axs_list__(array_map=array_map, orientation=orientation)

        return fig

    @classmethod
    def plot_ts(cls, obs_array, fmt="", orientation=None, dpi=150,
                fontsize=None, axs_fontsize=None, x_size=None, y_size=None,
                **kwargs):
        """
        Initialize a figure with the size best suited for the input data, then
        call plot() to plot time stream data.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            single time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param fmt: int, passed to plot()
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if 'horizontal', row will be spatial positional and
            column will be spectral position; if left None, class default
            'horizontal' will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 8
        :param float x_size: float, size of pixel in figure in inch
        :param float y_size: float
        :param kwargs: keywords to pass to plot()
        :return: FigArray object
        :rtype: FigArray
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, orientation=orientation,
                dpi=dpi, fontsize=fontsize, axs_fontsize=axs_fontsize,
                x_size=x_size, y_size=y_size)
        fig.plot(obs_array_plot, fmt, **kwargs)
        array_type = "tes" if isinstance(obs_array, ObsArray) else "mce"
        fig.set_labels(array_type=array_type, orientation=orientation)

        return fig

    @classmethod
    def plot_scatter(cls, obs_array, s=0.2, c=None, marker=".",
                     orientation=None, dpi=150, fontsize=None,
                     axs_fontsize=None, x_size=None, y_size=None, **kwargs):
        """
        Initialize a figure with the size best suited for the input data, then
        call scatter() to plot time stream data.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            single time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param float s: float, size of marker
        :param c: str or rgba tuple or array, if left None and data has
            chop, will plot color according to chop phase
        :type c: str or tuple or numpy.ndarray
        :param str marker: str, marker
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if 'horizontal', row will be spatial positional and
            column will be spectral position; if left None, class default
            'horizontal' will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 8
        :param float x_size: float, size of pixel in figure in inch
        :param float y_size: float
        :param kwargs: keywords to pass to scatter()
        :return: FigArray object
        :rtype: FigArray
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, orientation=orientation,
                dpi=dpi, fontsize=fontsize, axs_fontsize=axs_fontsize,
                x_size=x_size, y_size=y_size)
        fig.scatter(obs_array=obs_array_plot, s=s, c=c, marker=marker, **kwargs)
        array_type = "tes" if isinstance(obs_array, ObsArray) else "mce"
        fig.set_labels(array_type=array_type, orientation=orientation)

        return fig

    @classmethod
    def plot_errorbar(cls, obs_array, yerr=None, xerr=None, fmt=".", color=None,
                      orientation=None, dpi=150, fontsize=None,
                      axs_fontsize=None, x_size=None, y_size=None, **kwargs):
        """
        Initialize a figure with the size best suited for the input data, then
        call errorbar() to plot time stream data.

        :param obs_array: Obs or ObsArray, containing data to plot, must have a
            time axis, along which the data will be plotted
        :type obs_array: Obs or ObsArray
        :param yerr: Obs or ObsArray or DataObj or array, must have the same
            shape as obs_array
        :type yerr: Obs or ObsArray or DataObj or numpy.ndarray
        :param xerr: Obs or ObsArray or DataObj or array, must have the same
            shape as obs_array
        :type xerr: Obs or ObsArray or DataObj or numpy.ndarray
        :param str fmt: str, by default '.'
        :param color: str or tuple or array of rgba values
        :type color: str or tuple or numpy.ndarray
        :param str orientation: str, allowed are 'horizontal' and 'vertical', if
            'horizontal', row will be spatial positional and column will be
            spectral position; if left None, class default 'horizontal' will be
            used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 8
        :param float x_size: float, size of pixel in figure in inch
        :param float y_size: float
        :param kwargs: keywords to pass to errorbar()
        :return: FigArray object
        :rtype: FigArray
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, orientation=orientation,
                dpi=dpi, fontsize=fontsize, axs_fontsize=axs_fontsize,
                x_size=x_size, y_size=y_size)
        fig.errorbar(obs_array=obs_array, yerr=yerr, xerr=xerr, fmt=fmt,
                     color=color, **kwargs)
        array_type = "tes" if isinstance(obs_array, ObsArray) else "mce"
        fig.set_labels(array_type=array_type, orientation=orientation)

        return fig

    @classmethod
    def plot_psd(cls, obs_array, freq_ran=(0.5, 6), scale="linear",
                 orientation=None, dpi=150, fontsize=None,
                 axs_fontsize=None, x_size=None, y_size=None, **kwargs):
        """
        Initialize a figure with the size best suited for the input data, then
        call psd() to plot power spectral density.

        :param obs_array: Obs or ObsArray, containing data to do fft and the
            plot psd, must have a time axis, along which the data will be
            fourier transformed plotted
        :type obs_array: Obs or ObsArray
        :param freq_ran: tuple or list or array, the shown spectrum will be cut
            by freq_ran[0] <= freq <= freq_ran[1]
        :type freq_ran: tuple or list or numpy.ndarray
        :param str scale: str flag of the scale, only "linear", "log" and "dB"
            are accepted
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if 'horizontal', row will be spatial positional and
            column will be spectral position; if left None, class default
            'horizontal' will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 8
        :param float x_size: float, size of pixel in figure in inch
        :param float y_size: float
        :param kwargs: passed to psd()
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, orientation=orientation,
                dpi=dpi, fontsize=fontsize, axs_fontsize=axs_fontsize,
                x_size=x_size, y_size=y_size)
        fig.psd(obs_array=obs_array, scale=scale, freq_ran=freq_ran, **kwargs)
        array_type = "tes" if isinstance(obs_array, ObsArray) else "mce"
        fig.set_labels(array_type=array_type, orientation=orientation)

        return fig

    @classmethod
    def plot_specgram(cls, obs_array, nfft=5., noverlap=4., freq_ran=(0.5, 6),
                      scale="linear", orientation="horizontal", dpi=150,
                      fontsize=None, axs_fontsize=None, x_size=None,
                      y_size=None, **kwargs):
        """
        Initialize a figure with the size best suited for the input data, then
        call specgram to plot spectrogram for input obs_array.

        :param obs_array: Obs or ObsArray, object to calculate power spectral
            density and to plot, must have ts_ initialized
        :type obs_array: Obs or ObsArray
        :param nfft: number of data points in each block to do fft for int
            input, or seconds of data for each block for float input
        :type nfft: int or float
        :param noverlap: number of data points that overlap between blocks, or
            seconds of overlapping data points between blacks. If the value of
            noverlap >= nfft, then nfft number of data point - 1 will be used
        :type noverlap: int or float
        :param freq_ran: tuple or list or array, the shown spectrum will be cut
            by freq_ran[0] <= freq <= freq_ran[1]
        :type freq_ran: tuple or list or numpy.ndarray
        :param str scale: str flag of the scale, only "linear", "log" and "dB"
            are accepted
        :param str orientation: str, allowed values are 'horizontal' and
            'vertical', if 'horizontal', row will be spatial positional and
            column will be spectral position; if left None, class default
            'horizontal' will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of pixel axes, default 8
        :param float x_size: float, size of pixel in figure in inch
        :param float y_size: float
        :param kwargs: keyword arguments passed to specgram()
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, orientation=orientation,
                dpi=dpi, fontsize=fontsize, axs_fontsize=axs_fontsize,
                x_size=x_size, y_size=y_size)
        fig.specgram(obs_array=obs_array, nfft=nfft, noverlap=noverlap,
                     scale=scale, freq_ran=freq_ran, **kwargs)
        array_type = "tes" if isinstance(obs_array, ObsArray) else "mce"
        fig.set_labels(array_type=array_type, orientation=orientation)

        return fig


class FigSpec(FigFlux):
    """
    Plot spectrum of each spatial position
    """

    fontsize_ = 20  # type: int # fontsize of outer axes
    axs_fontsize_ = 8  # type: int # fontsize of axes for each spatial position
    x_size_, y_size_ = 0.2, 1.5  # type: float # size of each pixel axes in inch
    axs_list_ = None  # type: list
    twin_axs_list_ = None  # type: list
    spat_list_ = None  # type: list
    array_spat_llim_ = 0  # type: int
    array_spec_ulim_ = 0  # type: int

    def __init_axs_list__(self, array_map=None):
        if self.main_axes_ is None:
            extent = get_extent(
                    obs=array_map, orientation="horizontal", extent_offset=(
                        - 0.5 - 3 * self.axs_fontsize_ / 72 / self.x_size_,
                        - 0.5 + 3 * self.axs_fontsize_ / 72 / self.x_size_,
                        - 0.5 + 2 * self.axs_fontsize_ / 72 / self.y_size_,
                        - 0.5))
            self.__init_main_axes__(extent=extent)
        ax = self.main_axes_
        extent_round = (array_map.array_spec_llim_, array_map.array_spec_ulim_,
                        array_map.array_spat_llim_, array_map.array_spat_ulim_,)
        self.array_spec_llim_ = array_map.array_spec_llim_
        self.array_spec_ulim_ = array_map.array_spec_ulim_

        ax_x, ax_y = ax.get_position().bounds[:2]
        figsize = (self.get_figwidth(), self.get_figheight())
        dx = self.x_size_ * (extent_round[1] - extent_round[0] + 1) / figsize[0]
        x = ax_x + 3 * self.axs_fontsize_ / 72 / figsize[0]
        dy = self.y_size_ / figsize[1]
        y = 1 - self.fontsize_ * 4.6 / 72 / figsize[1] - dy

        axs_list, spat_list = [], []
        for spat in np.arange(extent_round[2], extent_round[3] + 1):
            axp = self.add_axes(
                    (x, y - (spat - extent_round[2]) * self.y_size_ / figsize[1],
                     dx, dy), zorder=10)
            axp.patch.set_alpha(0.4)
            axp.tick_params(axis="both", direction="in",
                            bottom=True, top=True, left=True, right=True,
                            labelbottom=False, labeltop=False)
            for item in ([axp.xaxis.label, axp.xaxis.offsetText] +
                         axp.get_xticklabels() +
                         [axp.yaxis.label, axp.yaxis.offsetText] +
                         axp.get_yticklabels()):
                item.set_fontsize(self.axs_fontsize_)
            axs_list.append(axp)
            spat_list.append(spat)
        axs_list[0].tick_params(labelbottom=False, labeltop=True)
        axs_list[-1].tick_params(labelbottom=True, labeltop=False)

        self.axs_list_ = axs_list
        self.spat_list_ = spat_list
        self.__set_ticks__()

    def __init_twin_axs_list__(self):
        if self.axs_list_ is None:
            raise ValueError("axs_list_ not initialized.")

        twin_axs_list = []
        for ax in self.axs_list_:
            twin_ax = ax.twinx()
            twin_ax.tick_params(axis="both", direction="in",
                                bottom=True, top=True, left=True, right=True)
            for item in ([twin_ax.yaxis.label, twin_ax.yaxis.offsetText] +
                         twin_ax.get_yticklabels()):
                item.set_fontsize(self.axs_fontsize_)
            twin_axs_list.append(twin_ax)
        self.twin_axs_list_ = twin_axs_list

    def __set_ticks__(self, xticks=False, yticks=True):
        super(FigSpec, self).__set_ticks__(xticks=xticks, yticks=yticks)
        if self.axs_list_ is not None:
            xlim = (self.array_spec_llim_ - 0.5, self.array_spec_ulim_ + 0.5)
            xticks = np.arange(int(np.ceil(xlim[0])), int(xlim[1]) + 1)
            for ax in self.axs_list_:
                ax.set_xticks(xticks)
                ax.set_xlim(xlim)
            ax = self.axs_list_[-1]
            ax.set_xticklabels(xticks)

    def set_xlim(self, xlim=None, twin_axes=False):
        """
        Set xlim for each spatial position if xlim is Obs/ObsArray object, or
        set for all axes if xlim is a tuple. if xlim is left None, then xlim
        will take the value of the minimum and maximum of xlim of all axes, to
        make all axes to share the same xlim.

        :param xlim: xlim to set for individual (Obs/ObsArray input) or all
            (tuple input) pixel axes. If left None, will pick the xlim suitable
            xlim all spatial axes
        :type xlim: tuple or list or Obs or ObsArray
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :raises ValueError: axes list not initialized
        :raises TypeError: invalid xlim type
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling xlim.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        if xlim is None:
            xlim_list = []
            for ax in axs_use:
                if len(ax.lines) + len(ax.collections) > 0:
                    xlim_list.append(ax.get_xlim())
            if len(xlim_list) == 0:
                xlim_list = [(0, 1)]
            xlim = (np.min(xlim_list), np.max(xlim_list))
        if isinstance(xlim, (tuple, list, np.ndarray)):
            for ax in axs_use:
                ax.set_xlim(xlim)
        elif isinstance(xlim, Obs):
            xlim = ObsArray(xlim)
            for ax, spat in zip(axs_use, self.spat_list_):
                xlim_spat = xlim.take_where(spat=spat)
                if not xlim_spat.empty_flag_:
                    lim = (np.nanmin(xlim_spat.data_),
                           np.nanmax(xlim_spat.data_))
                    ax.set_xlim(lim)
        else:
            raise TypeError("Invalid input type for xlim.")

    def set_ylim(self, ylim=None, twin_axes=False):
        """
        Set ylim on pixel base if ylim is Obs/ObsArray object, or set for all
        the pixels if ylim is a tuple. if ylim is left None, then ylim will take
        the value of the minimum and maximum of ylim of all axes, to make all
        axes to share the same ylim.

        :param ylim: ylim to set for individual (Obs/ObsArray input) or all
            (tuple input) pixel axes. If left None, will pick the ylim suitable
            ylim for all pixel axes
        :type ylim: tuple or list or Obs or ObsArray
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :raises ValueError: axes list not initialized
        :raises TypeError: invalid ylim type
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling xlim.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        if ylim is None:
            ylim_list = []
            for ax in axs_use:
                if len(ax.lines) + len(ax.collections) > 0:
                    ylim_list.append(ax.get_ylim())
            if len(ylim_list) == 0:
                ylim_list = [(0, 1)]
            ylim = (np.min(ylim_list), np.max(ylim_list))
        if isinstance(ylim, (tuple, list, np.ndarray)):
            for ax in axs_use:
                ax.set_ylim(ylim)
        elif isinstance(ylim, Obs):
            ylim = ObsArray(ylim)
            for ax, spat in zip(axs_use, self.spat_list_):
                ylim_spat = ylim.take_where(spat=spat)
                if 0 not in ylim_spat.shape_:
                    lim = (np.nanmin(ylim_spat.data_),
                           np.nanmax(ylim_spat.data_))
                    ax.set_ylim(lim)
        else:
            raise TypeError("Invalid input type for ylim.")

    def set_xlabel(self, xlabel, **kwargs):
        """
        set xlabel

        :param str xlabel: str
        :param kwargs: passed to axes.set_xlabel
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling xlabel.")
        ax = self.axs_list_[-1]
        ax.set_xlabel(xlabel, **kwargs)

    def set_ylabel(self, ylabel, twin_axes=False, **kwargs):
        """
        set xlabel

        :param str ylabel: str
        :param bool twin_axes: bool flag, whether to apply on the secondary
            y-axis
        :param kwargs: passed to axes.set_xlabel
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling ylabel.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        ax = axs_use[0]
        ax.set_ylabel(ylabel, **kwargs)

    def legend(self, twin_axes=False, *args, **kwargs):
        """
        Show legend on the uppermost axes of the leftmost column, with arg and
        kwargs passed to axes.legend()
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling legend.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        ax = axs_use[0]
        if "fontsize" not in kwargs:
            kwargs["fontsize"] = self.axs_fontsize_
        ax.legend(*args, **kwargs)

    def plot(self, obs_array, *args, mask=None, pix_flag_list=None,
             twin_axes=False, **kwargs):
        """
        plot input obs_array on the axes of obs_array.array_map_.array_spec_ on
        each spat, calling axes.plot()

        :param ObsArray obs_array: ObsArray, object to plot, should be length 1
            if ts is initialized
        :param args: arguments passed to axes.plot()
        :param numpy.ndarray mask: bool mask of flagged pixels, should have the
            same shape as obs_array
        :param list pix_flag_list: list of (spat, spec) of flagged pixels
        :param bool twin_axes: bool flag, whether to plot using the secondary
            y-axis
        :param kwargs: keyword arguments passed to axes.plot()
        """

        obs_array = ObsArray(arr_in=obs_array)
        if (obs_array.ts_.empty_flag_ and obs_array.ndim_ > 1 and
            obs_array.len_ > 1) or (not obs_array.ts_.empty_flag_ and
                                    ((obs_array.ndim_ > 2) or
                                     (obs_array.len_ > 1))):
            raise ValueError("Input obs_array should have length 1.")
        array_map = obs_array.array_map_
        if mask is None:
            mask = np.full(obs_array.shape_, fill_value=False, dtype=bool)
        mask_obs = ObsArray(arr_in=mask)
        mask_obs.update_array_map(array_map)
        pix_flag = array_map.get_flag_where(spat_spec_list=pix_flag_list)
        mask_obs.fill_by_flag_along_axis(pix_flag, axis=0, fill_value=True)

        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for spat, ax in zip(self.spat_list_, axs_use):
            array_map_use = array_map.take_where(spat=spat)
            if not array_map_use.empty_flag_:
                x, data = get_spat_data(
                        obs_array=obs_array, spat=spat, mask_obs=mask_obs)
                data = data.flatten()
                if np.count_nonzero(np.isfinite(data)) > 0:
                    ax.plot(x, data, *args, **kwargs)

    def step(self, obs_array, *args, where='mid', mask=None, pix_flag_list=None,
             twin_axes=False, **kwargs):
        """
        Step plot in each spat position of input obs_array.

        :param ObsArray obs_array: ObsArray, object to plot, should be length 1
            if ts is initialized
        :param args: arguments passed to axes.step()
        :param str where: string of step position passed to axes.step()
        :param numpy.ndarray mask: bool mask of flagged pixels, should have the
            same shape as obs_array
        :param list pix_flag_list: list of (spat, spec) of flagged pixels
        :param bool twin_axes: bool flag, whether to plot using the secondary
            y-axis
        :param kwargs: keyword arguments passed to axes.step()
        :raises ValueError: obs_array length > 1
        """

        obs_array = ObsArray(arr_in=obs_array)
        if (obs_array.ts_.empty_flag_ and obs_array.ndim_ > 1 and
            obs_array.len_ > 1) or (not obs_array.ts_.empty_flag_ and
                                    ((obs_array.ndim_ > 2) or
                                     (obs_array.len_ > 1))):
            raise ValueError("Input obs_array should have length 1.")
        array_map = obs_array.array_map_
        if mask is None:
            mask = np.full(obs_array.shape_, fill_value=False, dtype=bool)
        mask_obs = ObsArray(arr_in=mask, array_map=array_map)
        pix_flag = array_map.get_flag_where(spat_spec_list=pix_flag_list)
        mask_obs.fill_by_flag_along_axis(pix_flag, axis=0, fill_value=True)

        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_
        x = np.arange(array_map.array_spec_llim_,
                      array_map.array_spec_ulim_ + 1)

        for spat, ax in zip(self.spat_list_, axs_use):
            spec = np.full(len(x), fill_value=np.nan, dtype=float)
            array_map_use = array_map.take_where(spat=spat)
            if not array_map_use.empty_flag_:
                data_use = obs_array.take_by_array_map(array_map_use).data_. \
                    flatten()
                mask_use = mask_obs.take_by_array_map(array_map_use).data_. \
                    flatten()
                spec[array_map_use.array_spec_[~mask_use] -
                     array_map.array_spec_llim_] = data_use[~mask_use]
                if np.count_nonzero(np.isfinite(spec)) > 0:
                    ax.step(x, spec, *args, where=where, **kwargs)

    def errorbar(self, obs_array, yerr=None, xerr=None, fmt='.', color=None,
                 mask=None, pix_flag_list=None, twin_axes=False, **kwargs):
        """
        Errorbar plot

        :param ObsArray obs_array: ObsArray, object to plot, should be length 1
            if ts is initialized
        :param ObsArray yerr: ObsArray, must have the same shape as obs_array
        :param ObsArray xerr: ObsArray, must have the same shape as obs_array
        :param str fmt: str, by default '.'
        :param color: str or tuple or array of rgba
        :type color: str or tuple or numpy.ndarray
        :param numpy.ndarray mask: bool mask of flagged pixels, should have the
            same shape as obs_array
        :param list pix_flag_list: list of (spat, spec) of flagged pixels
        :param bool twin_axes: bool flag, whether to use the secondary y-axis
        :param kwargs: keywords to pass to axes.errorbar()
        """

        obs_array = ObsArray(arr_in=obs_array)
        if (obs_array.ts_.empty_flag_ and obs_array.ndim_ > 1 and
            obs_array.len_ > 1) or (not obs_array.ts_.empty_flag_ and
                                    ((obs_array.ndim_ > 2) or
                                     (obs_array.len_ > 1))):
            raise ValueError("Input obs_array should have length 1.")
        array_map = obs_array.array_map_
        xerr_obs = None if xerr is None else \
            ObsArray(arr_in=xerr).take_by_array_map(array_map_use=array_map)
        yerr_obs = None if yerr is None else \
            ObsArray(arr_in=yerr).take_by_array_map(array_map_use=array_map)
        if mask is None:
            mask = np.full(obs_array.shape_, fill_value=False, dtype=bool)
        mask_obs = ObsArray(arr_in=mask)
        mask_obs.update_array_map(array_map)
        pix_flag = array_map.get_flag_where(spat_spec_list=pix_flag_list)
        mask_obs.fill_by_flag_along_axis(pix_flag, axis=0, fill_value=True)

        if self.axs_list_ is None:
            self.__init_axs_list__(array_map=array_map)
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for spat, ax in zip(self.spat_list_, axs_use):
            array_map_use = array_map.take_where(spat=spat)
            if not array_map_use.empty_flag_:
                x, spec = get_spat_data(
                        obs_array=obs_array, spat=spat, mask_obs=mask_obs)
                spec = spec.flatten()
                if xerr_obs is not None:
                    (_, spec_xerr) = get_spat_data(
                            obs_array=xerr_obs, spat=spat, mask_obs=mask_obs)
                    spec_xerr = spec_xerr.flatten()
                else:
                    spec_xerr = None
                if yerr_obs is not None:
                    (_, spec_yerr) = get_spat_data(
                            obs_array=yerr_obs, spat=spat, mask_obs=mask_obs)
                    spec_yerr = spec_yerr.flatten()
                else:
                    spec_yerr = None
                if np.count_nonzero(np.isfinite(spec)) > 0:
                    ax.errorbar(x, spec, yerr=spec_yerr, xerr=spec_xerr,
                                fmt=fmt, color=color, **kwargs)

    def plot_all_spat(self, *args, twin_axes=False, **kwargs):
        """
        plot the same data in all the spatial position axes. All the args and
        kwargs inputs will be passed to axes.plot in every pixel axes.

        :param bool twin_axes: bool flag, whether to plot using the secondary
            y-axis
        """

        if self.axs_list_ is None:
            raise ValueError("Need to initialize spatial axes using ArrayMap " +
                             "before calling plot_all_pixels.")
        if twin_axes:
            if self.twin_axs_list_ is None:
                self.__init_twin_axs_list__()
            axs_use = self.twin_axs_list_
        else:
            axs_use = self.axs_list_

        for ax in axs_use:
            ax.plot(*args, **kwargs)

    @classmethod
    def init_by_array_map(cls, array_map, dpi=100, fontsize=None,
                          axs_fontsize=None, x_size=None, y_size=None,
                          **kwargs):
        """
        Initialize a figure with the size fit for the input array_map, with axes
        corresponding to the pixels in array_map

        :param ArrayMap array_map: ArrayMap of the array layout, the spat and
            spec information will be used
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of spatial position axes,
            default 8
        :param float x_size: float, size of pixel in figure in inch, default 0.2
        :param float y_size: float, default 1.5
        :param kwargs: keywords passed to plt.figure()
        :return: FigSpec object
        :rtype: FigSpec
        """

        array_map = ArrayMap(array_map) if not isinstance(array_map, Obs) else \
            ObsArray(arr_in=array_map).array_map_
        if fontsize is None:
            fontsize = cls.fontsize_
        if axs_fontsize is None:
            axs_fontsize = cls.axs_fontsize_
        if x_size is None:
            x_size = cls.x_size_
        if y_size is None:
            y_size = cls.y_size_

        extent = get_extent(
                obs=array_map, orientation="horizontal", extent_offset=(
                    - 0.5 - 3 * axs_fontsize / 72 / x_size,
                    - 0.5 + 3 * axs_fontsize / 72 / x_size,
                    - 0.5 + 2 * axs_fontsize / 72 / y_size,
                    - 0.5))
        fig = cls.init_by_extent(extent=extent, dpi=dpi, fontsize=fontsize,
                                 x_size=x_size, y_size=y_size, **kwargs)
        fig.axs_fontsize_ = axs_fontsize
        fig.__init_axs_list__(array_map=array_map)
        fig.set_labels(array_type="tes", orientation="horizontal",
                       x_labelpad=(fontsize if len(fig.axs_list_) > 1 else None))

        return fig

    @classmethod
    def plot_spec(cls, obs_array, yerr=None, xerr=None, dpi=100, fontsize=None,
                  axs_fontsize=None, x_size=None, y_size=None, color=None,
                  mask=None, pix_flag_list=None, **kwargs):
        """
        Initialize a figure with figsize suitable for the input data, then call
        step() first to make step plot, and call errorbar to make errorbar plot.
        kwargs will be passed to errorbar().

        :param ObsArray obs_array: ObsArray, object to plot, should be length 1
            if ts is initialized
        :param ObsArray yerr: ObsArray, must have the same shape as obs_array
        :param ObsArray xerr: ObsArray, must have the same shape as obs_array
        :param int dpi: int, dpi
        :param int fontsize: int, fontsize of outer frame, default 20
        :param int axs_fontsize: int, fontsize of spatial position axes,
            default 8
        :param float x_size: float, size of pixel in figure in inch, default 0.2
        :param float y_size: float, default 1.5
        :param color: str or tuple or array of rgba, passed to both step() and
            errorbar()
        :type color: str or tuple or numpy.ndarray
        :param numpy.ndarray mask: bool mask of flagged pixels, should have the
            same shape as obs_array
        :param list pix_flag_list: list of (spat, spec) of flagged pixels
        :param kwargs: keyword arguments passed to errorbar()
        :return: FigSpec object
        :rtype: FigSpec
        """

        obs_array_plot = ObsArray(arr_in=obs_array)
        fig = cls.init_by_array_map(
                array_map=obs_array_plot.array_map_, dpi=dpi, fontsize=fontsize,
                axs_fontsize=axs_fontsize, x_size=x_size, y_size=y_size)
        fig.step(obs_array=obs_array, color=color, mask=mask,
                 pix_flag_list=pix_flag_list)
        fig.imshow_flag(mask=mask, pix_flag_list=pix_flag_list)
        fig.errorbar(obs_array=obs_array, yerr=yerr, xerr=xerr, color=color,
                     mask=mask, pix_flag_list=pix_flag_list, **kwargs)
        fig.set_ylim(ylim=None)

        return fig


# TODO: class PlotSummary(PlotTS):

# ============================== helper functions ==============================


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


def get_extent(obs, orientation="horizontal",
               extent_offset=(-0.5, -0.5, -0.5, -0.5)):
    """
    Get the extent for imshow() for the input obs_array

    :param obs: Obs or ObsArray or array or ArrayMap, data to plot
    :type obs: Obs or ObsArray or numpy.ndarray or ArrayMap
    :param str orientation: str, allowed values are 'horizontal' and 'vertical'
    :param tuple extent_offset: tuple of offset value added to each value in
        extent, default -0.5 to center pixel
    :return extent: tuple, as the extent of the axes for imshow()
    :rtype: tuple
    """
    if isinstance(obs, (ObsArray, ArrayMap)):  # get shape of the input data
        array_map = obs.array_map_ if isinstance(obs, ObsArray) else obs
        if array_map.empty_flag_:
            raise ValueError("Empty ObsArray input.")
        extent = (array_map.array_spec_llim_, array_map.array_spec_ulim_ + 1,
                  array_map.array_spat_ulim_ + 1, array_map.array_spat_llim_,)
    else:
        data = DataObj(arr_in=obs)
        if data.empty_flag_:
            raise ValueError("Empty input.")
        arr = data.data_
        if arr.ndim == 1:
            extent = (0, 1, arr.shape[0], 0)
        else:
            extent = (0, arr.shape[1], arr.shape[0], 0)

    if orientation.lower().strip()[0] == "h":
        pass
    elif orientation.lower().strip()[0] == "v":
        extent = extent[::-1]
    else:
        raise ValueError("Invalid input orientation: %s." % orientation)

    extent = tuple([val + offset for (val, offset) in
                    zip(extent, extent_offset)])

    return extent


def get_extent_round(extent):
    """
    get an int tuple of inward rounded values of extent

    :param tuple extent: tuple, extent of axes

    :example:
    get_extent_round((-0.5, 1.5, 3.5, -0.5)) will return (0, 1, 0, 3)
    """
    return (int(np.ceil(min(extent[:2]))),
            int(np.ceil(max(extent[:2]) - 1)),
            int(np.ceil(min(extent[2:]))),
            int(np.ceil(max(extent[2:]) - 1)))


def get_corr_extent(extent):
    """
    Get the correct format of extent for pixel image which is offset by -0.5,
    and has y-axis flipped in order
    """
    extent_round = get_extent_round(extent)
    return (extent_round[0] - 0.5, extent_round[1] + 0.5,
            extent_round[3] + 0.5, extent_round[2] - 0.5)


def get_chop_color(chop):
    """
    Generate a rgba array based on chop phase, in which off chop is blue and
    on chop is red

    :param Chop chop: Chop
    """

    chop = Chop(chop)
    if not chop.empty_flag_:
        blue_arr = np.repeat([colors.to_rgba("blue")], chop.len_, axis=0)
        red_arr = np.repeat([colors.to_rgba("red")], chop.len_, axis=0)
        color = np.choose(chop.data_[..., None],
                          (blue_arr, red_arr))
    else:
        color = "black"

    return color


def get_spat_data(obs_array, spat, mask_obs=None):
    """
    return the spec and data recorded at the given spat
    """

    array_map_use = obs_array.array_map_.take_where(spat=spat).sort("spec")

    x = array_map_use.array_spec_
    data_use = obs_array.take_by_array_map(array_map_use).data_
    data = np.full(data_use.shape, fill_value=np.nan, dtype=float)

    if mask_obs is not None:
        mask_use = mask_obs.take_by_array_map(array_map_use).data_
        np.putmask(data, ~mask_use, data_use)
    else:
        data = data_use
    if data.ndim > 1:
        data = data.transpose()

    return x, data
