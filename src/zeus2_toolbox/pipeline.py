# @Date    : 2021-01-29 16:26:51
# @Credit  : Bo Peng(bp392@cornell.edu), Cody Lamarche, Christopher Rooney
# @Name    : pipeline.py
# @Version : 2.0
"""
Functions for the pipeline reduction.

requirements:
    beautifulsoup4, numpy >= 1.13, scipy
"""
import warnings

from .analyze import *
from .view import *

# define many default values used for the pipeline reduction
MATCH_SAME_PHASE = False  # default phase matching flag for stacking beam pairs,
# will match chop chunks of the opposite chop phase
STACK_FACTOR = 1  # default factor for the second beam in stacking beam pairs

MAD_THRE_BEAM = 7  # default MAD threshold for flagging ordinary observation
STD_THRE_FLAT = 2  # default MAD threshold for flagging skychop/flat
THRE_FLAG = 5E7  # absolute value threshold of the time series to flag a pixel
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
CHUNK_WEIGHT = True  # default flag for using weighted average of chunk when
# calculating flux

ORIENTATION = "horizontal"  # default orientation for making figure

NOD_PHASE = -1  # -1 means source is in on chop when beam is right, otherwise 1
NOD_COLNAME = "beam_is_R"  # column name in obs_info recording nodding phase

ZPOLD_SHAPE = (3, 3)  # default zpold shape, (az_len, elev_len) or (x_len, y_len)
ZPOLDBIG_SHAPE = (5, 5)  # default zpoldbig raster shape, (az_len, elev_len)
RASTER_THRE = 2  # default SNR requirement for not flagging a pixel

warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in less")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
warnings.filterwarnings(
        "ignore", message="Warning: converting a masked element to nan.")
warnings.filterwarnings(
        "ignore", message="From version 1.3 whiten='unit-variance' will " +
                          "be used by default.", category=FutureWarning)


# ================= intermediate level reduction functions =====================


def desnake_beam(obs, ref_pix=None, pix_flag_list=None, corr_thre=CORR_THRE,
                 min_pix_num=MIN_PIX_NUM, freq_sigma=FREQ_SIGMA,
                 edge_chunks_ncut=EDGE_CHUNKS_NCUT,
                 chunk_edges_ncut=CHUNK_EDGES_NCUT):
    """
    build a snake model for the input time series data in obs, fit and subtract
    the snake model for all the pixels; return the desnaked data, snake model
    for each pixel, amplitude of the snake model and the snake model

    :param obs: Obs or ObsArray object containing the time series
    :type obs: Obs or ObsArray
    :param list ref_pix: list of 2, [spat, spec] for ObsArray input or
        [row, col] for Obs input, the reference pixel to use in desnaking; will
        automatically determine the best correlated pixel if left None
    :param list or None pix_flag_list: list, [[spat, spec], ...] or
        [[row, col], ...] of the pixels not to include in building the snake,
        passed to stack_best_pixels()
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
    :rtype chunk_edges_ncut: int or float
    :return: list, (desnaked_obs, obs_snake, amp_snake, snake_model)
    :rtype: list
    """

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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


def ica_treat_beam(obs, spat_excl=None, pix_flag_list=None, verbose=VERBOSE,
                   finite_thre=FINITE_THRE, n_components_init=N_COMPONENTS_INIT,
                   n_components_max=N_COMPONENTS_MAX, max_iter=MAX_ITER,
                   random_state=RANDOM_STATE):
    """
    build noise model by running FastICA decomposition on each MCE column, then
    fit and subtract noise from the input data; return the data with noise
    subtracted, noise model for each pixel, noise feature amplitude for each
    pixel and noise features

    :param obs: Obs or ObsArray, object containing the time series
    :type obs: Obs or ObsArray
    :param list spat_excl: list, the spatial position range excluded in building
        noise model using ICA, e.g. spat_excl=[0, 2] means pixels at spat=0,1,2
        will not be used; will use all pixels if let None
    :param list or None pix_flag_list: list, [[spat, spec], ...] of pixels to
        exclude from ICA
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

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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
        obs_use = obs_flattened.take_where(col=col)
        if obs_use.shape_[0] > 0:
            ica = FastICA(
                    n_components=min(n_components_init, obs_use.shape_[0]),
                    fun='exp', max_iter=max_iter, random_state=random_state)
            obs_use = gaussian_filter_obs(
                    obs_use, freq_sigma=15, chunk_edges_ncut=4,
                    edge_chunks_ncut=1)
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


def plot_beam_ts(obs, title=None, pix_flag_list=None, reg_interest=None,
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
    :type obs: Obs or ObsArray or list or tuple or dict
    :param str title: str, title of the figure, will use the first available
        obs_id if left None
    :param list or None pix_flag_list: list, [[spat, spec], ...] or
        [[row, col], ...] of the flagged pixels, shown in grey shade
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

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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
        x_size = np.clip(obs_t_len.to(units.hour).to_value() / 2,
                         a_min=FigArray.x_size_, a_max=FigArray.x_size_ * 40)
    else:
        x_size = FigArray.x_size_
    array_map_use = array_map if reg_interest is None else \
        array_map.take_where(**reg_interest)
    x_len = array_map_use.array_spec_.max() - array_map_use.array_spec_.min() + \
            1 if check_orientation(orientation=orientation) else \
        array_map_use.array_spat_.max() - array_map_use.array_spat_.min() + 1
    x_size = min(x_size, 400 / x_len)

    fig = FigArray.init_by_array_map(array_map_use,
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


def plot_beam_flux(obs, title=None, pix_flag_list=None, plot_show=False,
                   plot_save=False, write_header=None, orientation=ORIENTATION):
    """
    plot flux for the pipeline reduction

    :param obs: Obs or ObsArray, the object containing the data to plot
    :type obs: Obs or ObsArray
    :param str title: str, title of the figure, will use the obs_id if left
        None
    :param list or None pix_flag_list: list, [[spat, spec], ...] or
        [[row, col], ...] of the flagged pixels, shown in grey shade
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

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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


def analyze_performance(beam, write_header=None, pix_flag_list=None, plot=False,
                        plot_rms=False, plot_ts=False, reg_interest=None,
                        plot_psd=True, plot_specgram=False, plot_show=False,
                        plot_save=False):
    """
    Analyze the performance of each pixel in the beam, including the rms of each
    pixel, plotting the time series, power spectral diagram (psd) and dynamical
    spectrum

    :param beam: Obs or ObsArray, with time series data
    :type beam: Obs or ObsArray
    :param str write_header: str, full path to the title to save files/figures,
        if left None, will write to current folder with obs_id as the title
    :param list or None pix_flag_list: list, a list including pixels to be flagged
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
    :rtype: Obs or ObsArray
    """

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), beam.obs_id_)
    beam_chop_rms = beam.chunk_proc(method="nanstd")
    beam_chop_rms.fill_by_mask(beam.chunk_proc(method="num_is_finite") <= 10)
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
                t_arr = time_to_gps_ts(Time(
                        np.char.replace(tb_use["UTC"], "U", "T"), format="isot"))
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
            # also plot log-log plot for the whole frequency range
            fig = FigArray.plot_psd(
                    beam if reg_interest is None else
                    ObsArray(beam).take_where(**reg_interest),
                    orientation=ORIENTATION, scale="dB", lw=0.5,
                    freq_ran=(0, 1 / beam.ts_.interv_ / 2))
            fig.imshow_flag(pix_flag_list=pix_flag_list, orientation=ORIENTATION)
            fig.set_xscale("log")
            fig.set_labels(beam, orientation=ORIENTATION)
            fig.set_title("%s log-log power spectral diagram" %
                          write_header.split("/")[-1])
            fig.set_ylabel("Spectral power [dB]")
            fig.set_xlabel("Frequency [Hz]")
            if plot_show:
                plt.show(fig)
            if plot_save:
                fig.savefig("%s_psd_log.png" % write_header)
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


def proc_beam(beam, write_header=None, is_flat=False, pix_flag_list=None, flat_flux=1,
              flat_err=0, cross=False, do_desnake=False, ref_pix=None,
              do_smooth=False, do_ica=False, spat_excl=None, do_clean=False,
              return_ts=False, return_pix_flag_list=False, plot=False,
              plot_ts=False, reg_interest=None, plot_flux=False, plot_show=False,
              plot_save=False, chunk_method=CHUNK_METHOD, method=METHOD):
    """
    process beam in the standard way, return chop flux, error and weight

    :param Obs or ObsArray beam: Obs or ObsArray object, with time series data
    :param str or None write_header: str, full path to the title to save
        files/figures, if left None, will write to current folder with {obs_id} as
        the title
    :param bool is_flat: bool, flag indicating this beam is flat field, which will
        use much larger mad flag threshold, flag pixel by SNR, and will not use
        weighted mean in calculating flux
    :param list or None pix_flag_list: list, a list including pixels to be flagged,
        will be combined with auto flagged pixels in making figures and in the
        returned pix_flag_list, the pixels will be flagged in the figure and the
        process of modelling noise
    :param flat_flux: Obs or ObsArray or scalar, the flat field flux to divide in
        computing the beam flux, must have the same shape and array_map; will
        ignore if is_flat is True; default is 1
    :type flat_flux: Obs or ObsArray or int or float
    :param flat_err: Obs or ObsArray or scalar, the flat field flux err used in
        computing the beam error, having the same behaviour as flat; default is 0
    :type flat_err: Obs or ObsArray or int or float
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
    :param bool do_clean: bool, [experimental] flag whether to clean data by
        filtering out 1+-0.015 Hz, 1.41+-0.015 Hz, 1.985+-0.015 Hz, 3.97+-0.03Hz,
        5.955+-0.045Hz, 7.94+-0.06Hz from the time series; if True, will happen
        after desnaking/smoother, before ICA
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

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    pix_flag_list += auto_flag_pix_by_ts(beam, thre_flag=THRE_FLAG)  # auto flag
    if (write_header is None) and plot:
        write_header = os.path.join(os.getcwd(), beam.obs_id_)
    beam_use, noise_beam = beam, type(beam)()
    plot_dict = {"raw data": beam}

    if do_desnake:  # model noise
        desnaked_beam, snake_beam, amp_snake, snake_model = desnake_beam(
                beam_use, ref_pix=ref_pix, pix_flag_list=pix_flag_list,
                corr_thre=CORR_THRE, min_pix_num=MIN_PIX_NUM,
                freq_sigma=FREQ_SIGMA, edge_chunks_ncut=EDGE_CHUNKS_NCUT,
                chunk_edges_ncut=CHUNK_EDGES_NCUT)
        beam_use = auto_flag_ts(desnaked_beam, is_flat=is_flat,
                                mad_thre=MAD_THRE_BEAM,
                                std_thre_flat=STD_THRE_FLAT)
        noise_beam += snake_beam
        plot_dict["snake"] = (noise_beam, {"c": "k"})
    if do_smooth:
        smooth_beam = gaussian_filter_obs(
                beam_use, freq_sigma=FREQ_SIGMA,
                edge_chunks_ncut=EDGE_CHUNKS_NCUT,
                chunk_edges_ncut=CHUNK_EDGES_NCUT).replace(chop=None)
        beam_use = auto_flag_ts(beam_use - smooth_beam, is_flat=is_flat,
                                mad_thre=MAD_THRE_BEAM,
                                std_thre_flat=STD_THRE_FLAT)
        noise_beam += smooth_beam
        plot_dict["smooth"] = (noise_beam, {"c": "y"})
    if do_clean:
        clean_beam = 0
        freq_center_list = (1.41, 1.985, 3.97, 5.955, 7.94)
        freq_sigma_list = (0.015, 0.015, 0.03, 0.045, 0.06)
        if not is_flat:
            freq_center_list += (.9, 1.,)
            freq_sigma_list += (0.02, 0.02)
        for freq_center, freq_sigma in zip(freq_center_list, freq_sigma_list):
            clean_beam += (gaussian_filter_obs(
                    beam_use, freq_sigma=freq_sigma, freq_center=freq_center,
                    edge_chunks_ncut=0, chunk_edges_ncut=0, truncate=3.0) +
                           gaussian_filter_obs(
                                   beam_use, freq_sigma=freq_sigma, freq_center=-freq_center,
                                   edge_chunks_ncut=0, chunk_edges_ncut=0, truncate=3.0)) / 2
        beam_use = auto_flag_ts(beam_use - clean_beam, is_flat=is_flat,
                                mad_thre=MAD_THRE_BEAM,
                                std_thre_flat=STD_THRE_FLAT)
        noise_beam += clean_beam
        plot_dict["clean"] = (noise_beam, {"c": "c"}) if \
            ("snake" in plot_dict) or ("smooth" in plot_dict) else \
            (noise_beam, {"c": "c", "twin_axes": True})
    if do_ica:
        ica_treated_beam, ica_noise_beam, amp_ica, ica_noise_model = \
            ica_treat_beam(beam_use, spat_excl=spat_excl,
                           pix_flag_list=pix_flag_list, verbose=VERBOSE,
                           finite_thre=FINITE_THRE,
                           n_components_init=N_COMPONENTS_INIT,
                           n_components_max=N_COMPONENTS_MAX, max_iter=MAX_ITER,
                           random_state=RANDOM_STATE)
        beam_use = auto_flag_ts(ica_treated_beam, is_flat=is_flat,
                                mad_thre=MAD_THRE_BEAM,
                                std_thre_flat=STD_THRE_FLAT)
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
                None if (is_flat or not CHUNK_WEIGHT) else
                1 / (beam_use.chunk_proc("nanstd", keep_shape=True) ** 2 /
                     beam_use.chunk_proc("num_is_finite", keep_shape=True) +
                     (beam_use.chunk_proc("nanmedian", keep_shape=True) -
                      beam_use.chunk_proc("nanmean", keep_shape=True)) ** 2)),
            err_type="external")
    pix_flag_list = auto_flag_pix_by_flux(  # auto flag
            beam_flux, beam_err, pix_flag_list=pix_flag_list, is_flat=is_flat,
            snr_thre=SNR_THRE, mad_thre_err=MAD_THRE_BEAM_ERR)
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


def make_raster(beams_flux, beams_err=None, write_header=None, pix_flag_list=None,
                raster_shape=ZPOLDBIG_SHAPE, return_pix_flag_list=False,
                plot=False, reg_interest=None, plot_show=False, plot_save=False,
                raster_thre=RASTER_THRE):
    """
    format the last dimension of the input object recording beams flux into 2-d
    raster, then plot the raster; the raster always starts from the lower left,
    then swipe to the right, move upwards by one, swiping to the left and so on,
    zigzagging to the top row.

    :param beams_flux: Obs or ObsArray, with flux of all the beams in the last
        dimension
    :type beams_flux: Obs or ObsArray
    :param beams_err: Obs or ObsArray, with error of all the beams in the last
        dimension, used for flagging pixels
    :type beams_err: Obs or ObsArray
    :param str write_header: str, full path to the title to save figures,
        if left None, will write to current folder with {obs_id} as the title
    :param list or None pix_flag_list: list, a list including pixels to be flagged,
        will be combined with auto flagged pixels in making figures and in the
        returned pix_flag_list, additional pixel flagging will be performed based
        on S/N
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
    :type raster_thre: int or float
    :return: tuple (raster_flux, [pix_flag_list]), are (ObsArray recording the
        raster flux reshaped in the last two dimensions. [optional, list of auto
        flagged pixels])
    :rtype: tuple
    """

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    beams_flux_obs_array = ObsArray(beams_flux.copy())
    array_map = beams_flux_obs_array.array_map_
    raster_len = np.prod(raster_shape, dtype=int)
    x_len, y_len = raster_shape
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
        pix_flag_list += array_map.take_by_flag(
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

    raster_flux = beams_flux_obs_array.replace(arr_in=np.where(  # rearrange shape
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
                x_size=0.5, y_size=0.5, axs_fontsize=2)
        fig.imshow(raster_flux, origin="lower",
                   extent=(-x_len / 2, x_len / 2, -y_len / 2, y_len / 2))
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

        result_list, text_list = [], []  # fit gaussian to raster
        gauss_2d_fit = lambda pos, x0, y0, sigma, amp, offset: \
            gaussian_2d(pos, x0=x0, y0=y0, sigma_x=sigma, sigma_y=sigma,
                        theta=0, amp=amp) + offset
        xx, yy = np.meshgrid(np.arange(x_len, dtype=float),
                             np.arange(y_len, dtype=float))
        xx -= (x_len - 1) / 2
        yy -= (y_len - 1) / 2
        raster_fit = raster_flux if reg_interest is None else \
            raster_flux.take_where(**reg_interest)
        for pix_raster in raster_fit.data_:
            finite_mask = np.isfinite(pix_raster)
            if finite_mask.sum() >= 5:
                try:
                    result = curve_fit(
                            gauss_2d_fit, xdata=(xx[finite_mask], yy[finite_mask]),
                            ydata=pix_raster[finite_mask], p0=(0, 0, 1, 1, 0))
                    text_list += [["x0=%.2f\ny0=%.2f\n sigma=%.1f" %
                                   tuple(result[0][:3])]]
                except RuntimeError:
                    text_list += [["fitting\nfailed"]]
            else:
                text_list += [["not\nenough\ndata"]]

        fig = FigArray.write_text(
                raster_fit.replace(arr_in=text_list), orientation=ORIENTATION,
                x_size=0.5, y_size=0.5, text_fontsize=6)
        fig.imshow_flag(pix_flag_list=pix_flag_list)
        fig.set_title(title="%s raster fit" % write_header.split("/")[-1])
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig("%s_raster_fit.png" % write_header)
        plt.close(fig)

    result = (raster_flux,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def stack_raster(raster, raster_wt=None, write_header=None, pix_flag_list=None,
                 plot=False, plot_show=False, plot_save=False):
    """
    Stack the raster along spatial dimension to get a high SNR raster, used for
    analyzing data taken with zpold or zpoldbig

    :param ObsArray raster: ObsArray, object containing the raster in the last
        two dimensions
    :param raster_wt: weight of the raster, raster * raster_wt will be stacked
    :type raster_wt: int or float or numpy.ndarray or Obs or ObsArray
    :param str write_header: str, full path to the title to save files/figures,
        if left None, will write to current folder with {obs_id}_raster_stack
        as file header
    :param list or None pix_flag_list: list, a list including pixels to be flagged,
        these pixels will not be used in stacking
    :param bool plot: bool, flag whether to make the figure of the stacked raster
    :param bool plot_show: bool, flag whether to show the figure
    :param bool plot_save: bool, flag whether to save the figure
    :return: ObsArray object containing the stacked raster
    :rtype: ObsArray
    """

    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    raster_norm = ObsArray(raster * raster_wt)
    y_len, x_len = raster.shape_[-2:]
    raster_norm -= raster_norm.proc_along_axis("nanmedian", axis=-1). \
        proc_along_axis("nanmedian", axis=-2)
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
        fig = FigArray.init_by_array_map(
                stacked_raster, orientation=ORIENTATION,
                x_size=FigArray.y_size_)
        fig.imshow(stacked_raster, origin="lower",
                   extent=(-x_len / 2, x_len / 2, -y_len / 2, y_len / 2))
        fig.set_xlabel("azimuth")
        fig.set_ylabel("altitude")
        fig.set_labels(raster, orientation=ORIENTATION)
        fig.set_title(title="%s\n stacked raster" % write_header.split("/")[-1])
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig("%s_raster_stack.png" % write_header)
        plt.close(fig)

        result_list, text_list = [], []  # fit gaussian to raster
        gauss_2d_fit = lambda pos, x0, y0, sigma, amp, offset: \
            gaussian_2d(pos, x0=x0, y0=y0, sigma_x=sigma, sigma_y=sigma,
                        theta=0, amp=amp) + offset
        xx, yy = np.meshgrid(np.arange(x_len, dtype=float),
                             np.arange(y_len, dtype=float))
        xx -= (x_len - 1) / 2
        yy -= (y_len - 1) / 2
        for pix_raster in stacked_raster.data_:
            finite_mask = np.isfinite(pix_raster)
            if finite_mask.sum() >= 5:
                try:
                    result = curve_fit(
                            gauss_2d_fit, xdata=(xx[finite_mask], yy[finite_mask]),
                            ydata=pix_raster[finite_mask], p0=(0, 0, 1, 1, 0))
                    text_list += [["x0=%.2f\ny0=%.2f\n sigma=%.1f" %
                                   tuple(result[0][:3])]]
                except RuntimeError:
                    text_list += [["fitting\nfailed"]]
            else:
                text_list += [["not\nenough\ndata"]]

        fig = FigArray.write_text(
                stacked_raster.replace(arr_in=text_list),
                orientation=ORIENTATION, text_fontsize=20)
        fig.set_title(title="%s\n stacked raster fit" %
                            write_header.split("/")[-1])
        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig("%s_raster_stack_fit.png" % write_header)
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
    :rtype: Obs or ObsArray
    """

    try:
        beam = Obs.read_header(filename=file_header)  # read in data
    except Exception as err:
        warnings.warn("fail to read in %s due to %s: %s" %
                      (file_header, type(err), err), UserWarning)
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
        beam = auto_flag_ts(beam, is_flat=is_flat, mad_thre=MAD_THRE_BEAM,
                            std_thre_flat=STD_THRE_FLAT)
    if (obs_log is not None) and (len(obs_log) > 0) and (not beam.empty_flag_):
        with warnings.catch_warnings():
            if is_flat:
                warnings.filterwarnings(
                        "ignore", message="No entry is found in obs log.")
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
    :type stack_factor: int or float
    :return: Obs or ObsArray object containing the stacked beam pair
    :rtype: Obs or ObsArray
    """

    beam1, beam2 = [read_beam(file_header, array_map=array_map, obs_log=obs_log,
                              flag_ts=flag_ts, is_flat=is_flat)
                    for file_header in (file_header1, file_header2)]
    matched_beam1, matched_beam2 = get_match_phase_obs(
            beam1, beam2, match_same_phase=match_same_phase)
    stacked_beam_pair = (matched_beam1 + stack_factor * matched_beam2) / 2
    if flag_ts:
        stacked_beam_pair = auto_flag_ts(stacked_beam_pair, is_flat=is_flat,
                                         mad_thre=MAD_THRE_BEAM,
                                         std_thre_flat=STD_THRE_FLAT)

    return stacked_beam_pair


def read_tp(file_header, array_map=None, obs_log=None, flag_ts=True,
            is_flat=False, t0=None, freq=None):
    """
    function to read total power data which does not come with .ts file, so the
    time series is reconstructed with the best guess using T0=CTIME+19.5s and
    data sampling frequency in MCE data header

    :param str file_header: str, full path to the data file
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    :param ObsLog obs_log: ObsLog, optional, if not None, will try to find the
        entry in the provided obs_log and add to the obs_info of the output obj
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :param float t0: float, starting time of the time series, if left as None,
        will try to use ObsInfo.table_["CTIME"]+19.5, and fallback to 0.0 if the
        "CTIME" column doesn't exist
    :param float freq: float, data sampling rate of the time series, if left as
        None, will try to use ObsInfo.table_["freq"], and fallback to
        398.72408293460927 if the "freq" column doesn't exist
    :return: Obs or ObsArray object containing the data
    :rtype: Obs or ObsArray
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", message="%s not found." % (file_header + ".ts"))
        beam = read_beam(file_header=file_header, array_map=array_map,
                         obs_log=None, flag_ts=flag_ts, is_flat=is_flat)

    if beam.ts_.empty_flag_:
        if t0 is None:
            t0 = 0.0
            if not beam.obs_info_.empty_flag_:
                if "CTIME" in beam.obs_info_.colnames_:
                    t0 = time_to_gps_ts(Time(
                            beam.obs_info_.table_["CTIME"][0], format="unix"))
                else:
                    warnings.warn("can not find CTIME, using default value.",
                                  UserWarning)
        if freq is None:
            freq = 398.72408293460927
            if "freq" in beam.obs_info_.colnames_:
                dt = beam.obs_info_.table_["freq"][0]
            else:
                warnings.warn("can not find freq, using default value.",
                              UserWarning)
        ts = np.arange(beam.len_, dtype=np.double) / freq + t0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Time = 0 exists in TimeStamp.")
            beam.update_ts(ts)

    if (obs_log is not None) and (len(obs_log) > 0) and (not beam.empty_flag_):
        with warnings.catch_warnings():
            if is_flat:
                warnings.filterwarnings(
                        "ignore", message="No entry is found in obs log.")
            beam.match_obs_log(obs_log)  # find entry in obs_log

    return beam


def read_iv_curve(file_header, array_map=None):
    """
    function to read IV curve file, using .bias as the tie series

    :param str file_header: str, full path to the data file
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "%s.chop not found." % file_header)
        warnings.filterwarnings("ignore", "%s.ts not found." % file_header)
        warnings.filterwarnings(
                "ignore", ("Failed to read .hk for %s due to " % file_header) +
                          ("<class 'FileNotFoundError'>: %s or %s.hk " %
                           (file_header, file_header)) + "are not hk files.")
        beam = read_beam(file_header=file_header, array_map=array_map,
                         obs_log=None, flag_ts=False, is_flat=False)
    bias = Tb.read(file_header + ".bias", format="ascii.csv")["<tes_bias>"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Time = 0 exists in TimeStamp.")
        beam.update_ts(bias)

    return beam


def read_bias_step(file_header, array_map=None, flag_ts=True, is_flat=False,
                   t0=None, freq=None, data_rate=None, ramp_step_period=None):
    """
    function to read total power data which does not come with .ts file, so the
    time series is reconstructed with the best guess using T0=CTIME+19.5s and
    data sampling frequency in MCE data header. Caution: the first data point is
    dumped

    :param str file_header: str, full path to the data file
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :param float t0: float, starting time of the time series, if left as None,
        will try to use ObsInfo.table_["CTIME"]+19.5, and fallback to 0.0 if the
        "CTIME" column doesn't exist
    :param float freq: float, data sampling rate of the time series, if left as
        None, will try to use ObsInfo.table_["freq"], and fallback to
        398.72408293460927 if the "freq" column doesn't exist
    :param int or float data_rate: int or float, data sampling rate in the MCE
        clock unit, ramp_step_period/data_rate is the bias step rate, if left
        None will try to use ObsInfo.table_["data_rate"], then try
        ObsInfo.table_["RB_cc_data_rate"], and fallback to 38 if neither exist
    :param int or float ramp_step_period: int or float, bias ramping period in MCE
        clock unit, if left None will try to use
        ObsInfo.table_["RB_cc_ramp_step_period"], and fallback to 3800 if
        the column RB_cc_ramp_step_period is not found
    :return: Obs or ObsArray object containing the data
    :rtype: Obs or ObsArray
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "%s not found." % (file_header + ".chop"))
        beam = read_tp(file_header=file_header, array_map=array_map, obs_log=None,
                       flag_ts=flag_ts, is_flat=is_flat, t0=t0, freq=freq)
    beam = beam.replace(arr_in=beam.data_[..., 1:])

    if data_rate is None:
        data_rate = 38
        if "data_rate" in beam.obs_info_.colnames_:
            data_rate = beam.obs_info_.table_["data_rate"][0]
        elif "RB_cc_data_rate" in beam.obs_info_.colnames_:
            data_rate = beam.obs_info_.table_["RB_cc_data_rate"][0]
        else:
            warnings.warn("can not find data_rate, using default value.",
                          UserWarning)
    if ramp_step_period is None:
        ramp_step_period = 38 * 199
        if "RB_cc_ramp_step_period" in beam.obs_info_.colnames_:
            ramp_step_period = beam.obs_info_.table_["RB_cc_ramp_step_period"][0]
        else:
            warnings.warn("can not find ramp_step_period, using default value.",
                          UserWarning)
    ramp_chop = ((np.arange(beam.len_) * data_rate // ramp_step_period) % 2)
    beam.update_chop(ramp_chop.astype(bool))

    return beam


def reduce_beam(file_header, write_dir=None, write_suffix="", array_map=None,
                obs_log=None, is_flat=False, pix_flag_list=None, flat_flux=1,
                flat_err=0, cross=False, do_desnake=False, ref_pix=None,
                do_smooth=False, do_ica=False, spat_excl=None, do_clean=False,
                return_ts=False,
                return_pix_flag_list=False, plot=False, plot_ts=False,
                reg_interest=None, plot_flux=False, plot_show=False,
                plot_save=False):
    """
    a wrapper function to read data and reduce beam in the standard way
    """

    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    write_suffix = str(write_suffix)
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_clean, do_ica),
                            ("desnake", "smooth", "clean", "ica")):
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
                do_clean=do_clean,
                return_ts=return_ts, return_pix_flag_list=return_pix_flag_list,
                plot=plot, plot_ts=plot_ts, reg_interest=reg_interest,
                plot_flux=plot_flux, plot_show=plot_show, plot_save=plot_save,
                chunk_method=CHUNK_METHOD, method=METHOD)
    else:
        result = (beam.copy(), beam.copy(), beam.copy())
        if return_ts:
            result += (beam,)
        if return_pix_flag_list:
            result += (pix_flag_list,)

    return result


def reduce_beam_pair(file_header1, file_header2, write_dir=None, write_suffix="",
                     array_map=None, obs_log=None, is_flat=False,
                     pix_flag_list=None, flat_flux=1, flat_err=0, do_desnake=False,
                     ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                     do_clean=False,
                     return_ts=False, return_pix_flag_list=False, plot=False,
                     plot_ts=False, reg_interest=None, plot_flux=False,
                     plot_show=False, plot_save=False):
    """
    a wrapper function to read data and reduce beam pair in the standard way
    """

    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    write_suffix = str(write_suffix)
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_clean, do_ica),
                            ("desnake", "smooth", "clean", "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method

    print("Processing beam pair %s and %s." % (file_header1, file_header2))
    beam_pair = read_beam_pair(file_header1=file_header1, file_header2=file_header2,
                               array_map=array_map, obs_log=obs_log,
                               flag_ts=True, is_flat=is_flat,
                               match_same_phase=MATCH_SAME_PHASE,
                               stack_factor=STACK_FACTOR)
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
                do_desnake=do_desnake, ref_pix=ref_pix, do_smooth=do_smooth,
                do_ica=do_ica, spat_excl=spat_excl, do_clean=do_clean,
                return_ts=return_ts,
                return_pix_flag_list=return_pix_flag_list, plot=plot,
                plot_ts=plot_ts, reg_interest=reg_interest, plot_flux=plot_flux,
                plot_show=plot_show, plot_save=plot_save,
                chunk_method=CHUNK_METHOD, method=METHOD)
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

    :param list file_header_list: list, of the paths to the headers of the data
    :param ArrayMap array_map: ArrayMap, optional, if not None, will transform
        flat data into ObsArray and then process
    :param ObsLog obs_log: ObsLog, optional, if not None, will try to find the
        entry in the provided obs_log and add to the obs_info of the output obj
    :param bool flag_ts: bool, flag whether to flag outliers in time series by
        auto_flag_ts(), default True
    :param bool is_flat: bool, flag whether the beam is flat/skychop, passed to
        auto_flag_ts(), default False
    :param bool parallel: bool, flag whether to run it in parallelized mode,
        would accelerate the process by many factors on a multicore machine
    :return: Obs or ObsArray object containing all the data concatenated
    :rtype: Obs or ObsArray
    """

    args_list = []  # build variable list for read_beam
    for file_header in file_header_list:
        args = ()
        for var_name in inspect.getfullargspec(read_beam)[0]:
            args += (locals()[var_name],)
        args_list.append(args)

    if parallel and check_parallel() and len(args_list) > 1:
        results = parallel_run(func=read_beam, args_list=args_list)
    else:
        results = []
        for args in args_list:
            results.append(read_beam(*args))

    kwargs = {}
    if array_map is not None:  # combine all beams
        type_result = ObsArray
        kwargs["array_map"] = array_map
    else:
        type_result = Obs
    data_list, ts_list, chop_list = [], [], []
    obs_id_list, obs_id_arr_list, obs_info_list = [], [], []
    for beam in results:
        if not beam.empty_flag_:
            data_list.append(beam.data_)
            ts_list.append(beam.ts_.data_)
            chop_list.append(beam.chop_.data_)
            obs_id_list += beam.obs_id_list_
            obs_id_arr_list.append(beam.obs_id_arr_.data_)
            obs_info_list.append(beam.obs_info_.table_)
    kwargs["arr_in"] = np.concatenate(data_list, axis=-1)
    del data_list
    kwargs["ts"] = np.concatenate(ts_list)
    del ts_list
    kwargs["chop"] = np.concatenate(chop_list)
    del chop_list
    kwargs["obs_id_list"] = obs_id_list
    kwargs["obs_id_arr"] = np.concatenate(obs_id_arr_list)
    del obs_id_arr_list
    kwargs["obs_info"] = vstack_reconcile(obs_info_list, join_type="outer")
    del obs_info_list
    kwargs["obs_id"] = obs_id_list[0]
    all_beams = type_result(**kwargs)

    return all_beams


# ====================== high level reduction functions ========================


def reduce_beams(data_header, data_dir=None, write_dir=None, write_suffix="",
                 array_map=None, obs_log=None, is_flat=False, pix_flag_list=None,
                 flat_flux=1, flat_err=0, cross=False, parallel=False,
                 do_desnake=False, ref_pix=None, do_smooth=False, do_ica=False,
                 spat_excl=None, do_clean=False, return_ts=False,
                 return_pix_flag_list=False,
                 plot=False, plot_ts=False, reg_interest=None, plot_flux=False,
                 plot_show=False, plot_save=False, **kwargs):
    """
    reduce the data of beam in data_header, and return the flux of beams
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    args_list = []  # build variable list for reduce_beam()
    flat_flux_group_flag, flat_err_group_flag = False, False
    if isinstance(flat_flux, (Obs, ObsArray)) and flat_flux.len_ > 1:
        flat_flux_group_flag, flat_flux_group = True, flat_flux.copy()
    if isinstance(flat_err, (Obs, ObsArray)) and flat_err.len_ > 1:
        flat_err_group_flag, flat_err_group = True, flat_err.copy()
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

    if parallel and check_parallel() and len(args_list) > 1:
        results = parallel_run(func=reduce_beam, args_list=args_list)
    else:
        results = []
        for args in args_list:
            results.append(reduce_beam(*args))

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
                      is_flat=False, pix_flag_list=None, flat_flux=1, flat_err=0,
                      parallel=False, do_desnake=False, ref_pix=None,
                      do_smooth=False, do_ica=False, spat_excl=None,
                      do_clean=False,
                      return_ts=False, return_pix_flag_list=False, plot=False,
                      plot_ts=False, reg_interest=None, plot_flux=False,
                      plot_show=False, plot_save=False, use_hk=True, **kwargs):
    """
    reduce the data files in data_header by calling reduce_beam_pair() which
    stack each beam pair, and return the flux, error and weight of the beam pairs

    :raises RunTimeError: no beam pair is matched
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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
            del beams_info, beams_left, beams_right
        if len(args_list) == 0:
            raise RuntimeError("No beam pair is matched, may not be nodding.")

    if parallel and check_parallel() and len(args_list) > 1:
        results = parallel_run(func=reduce_beam_pair, args_list=args_list)
    else:
        results = []
        for args in args_list:
            results.append(reduce_beam_pair(*args))

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
                   array_map=None, obs_log=None, pix_flag_list=None,
                   parallel=False,
                   return_ts=False, return_pix_flag_list=True, table_save=True,
                   plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                   plot_show=False, plot_save=True, analyze=False):
    """
    process data taken as skychop
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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
                flat_beams_ts,
                write_header=os.path.join(write_dir, flat_file_header),
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
                array_map=None, obs_log=None, pix_flag_list=None, flat_flux=1,
                flat_err=0, parallel=False, stack=False, do_desnake=False,
                ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                do_clean=False, return_ts=False, return_pix_flag_list=True,
                table_save=True, save_wl=True, save_atm=True, plot=True,
                plot_ts=True, plot_atm=True, reg_interest=None,
                plot_flux=True, plot_show=False, plot_save=True, analyze=False,
                use_hk=True, grat_idx=None, pwv=None, elev=None):
    """
    reduce the data from zobs command

    :param bool use_hk: bool, flag whether to use hk file as nodding phase
    :param bool plot_atm: bool, whether to try to plot atmospheric transmission,
        ignored if plot=False
    :param bool save_wl: bool, whether to try to convert the array to wavelength
        and save to table, ignored if table_save=False
    :param bool save_atm: bool, whether to try to compute the atmospheric
        transmission and save to table, ignored if table_save=False
    :param int or float grat_idx: int or float, the grating index to convert the
        array map into wavelength, with the priority grat_idx > obs_log >
        array_map.conf, if any of them have non-zero valid values
    :param int or float pwv: int or float, PWV in mm for computing the atmospheric
        transmission, with the priority pwv > obs_log > array_map.conf, if
        any of them have non-zero valid values
    :param int or float elev: int or float, telescope elevation in degree for
        computing the atmospheric transmission, with the priority elev >
        obs_log > array_map.conf, if any of them have non-zero valid values

    :raises RunTimeError: not nodding
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_clean, do_ica),
                            ("desnake", "smooth", "clean", "ica")):
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
                spat_excl=spat_excl, do_clean=do_clean,
                return_ts=return_ts | analyze,
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
                spat_excl=spat_excl, do_clean=do_clean,
                return_ts=return_ts | analyze,
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
    # internal error of the weighted mean
    zobs_err_in = \
        1 / (1 / beam_pairs_err ** 2).proc_along_time("nansum").sqrt() / \
        beam_pairs_flux.proc_along_time("num_is_finite").sqrt()
    zobs_err = zobs_err_ex.replace(
            arr_in=np.choose(zobs_err_ex.data_ < zobs_err_in.data_,
                             [zobs_err_ex.data_, zobs_err_in.data_]))
    pix_flag_list = auto_flag_pix_by_flux(zobs_flux, zobs_err,
                                          pix_flag_list=pix_flag_list,
                                          snr_thre=SNR_THRE,
                                          mad_thre_err=MAD_THRE_BEAM_ERR)

    # convert array map to wavelength
    if (array_map is not None) and \
            ((table_save and save_wl) or
             ((table_save or plot) and (plot_atm or save_atm))):
        grat_idx = configure_helper(zobs_flux, keyword=("grat_idx", "gratingindex"),
                                    var=grat_idx, supersede=True)
        if is_meaningful(grat_idx):
            array_map.init_wl(grat_idx=grat_idx)

    trans_flag = False  # compute transmission
    if (array_map is not None) and array_map.wl_flag_ and \
            (table_save or plot) and (plot_atm or save_atm):
        pwv = configure_helper(zobs_flux, keyword=("pwv", "mm PWV"),
                               var=pwv, supersede=True)
        elev = configure_helper(zobs_flux, keyword=("elev", "Elevation"),
                                var=elev, supersede=True)
        if is_meaningful(pwv) and is_meaningful(elev):
            atm_trans_raw = get_transmission_raw_obs_array(
                    array_map=array_map, pwv=pwv, elev=elev, grat_idx=grat_idx)
            atm_trans = get_transmission_obs_array(
                    array_map=array_map, pwv=pwv, elev=elev)
            atm_trans.replace(ts=zobs_flux.ts_, obs_id=zobs_flux.obs_id_,
                              obs_id_arr=zobs_flux.obs_id_arr_)
            trans_flag = True

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
        if save_wl and (array_map is not None) and array_map.wl_flag_:
            tb_list += [zobs_flux.replace(arr_in=array_map.array_wl_[:, None]),
                        zobs_flux.replace(arr_in=array_map.array_d_wl_[:, None])]
            tb_names += ["wl", "d_wl"]
        if save_atm and trans_flag:
            tb_list += [atm_trans, ]
            tb_names += ["atm_trans", ]
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
                        write_dir, "%s_flux" % data_file_header),
                orientation=ORIENTATION))
        plt.close(plot_beam_flux(
                zobs_err, title=data_file_header + " zobs error",
                pix_flag_list=pix_flag_list, plot_show=plot_show,
                plot_save=plot_save, write_header=os.path.join(
                        write_dir, "%s_err" % data_file_header),
                orientation=ORIENTATION))

        # plot spectrum
        fig = FigSpec.plot_spec(zobs_flux, yerr=zobs_err,
                                pix_flag_list=pix_flag_list, color="k")
        fig.imshow_flag(pix_flag_list=pix_flag_list)
        array_map_tmp = ObsArray(zobs_flux).array_map_
        fig.plot_all_spat([array_map_tmp.array_spec_llim_,
                           array_map_tmp.array_spec_ulim_], [0, 0], "k:")
        if plot_atm and trans_flag:
            fig.plot(atm_trans_raw, "c:", twin_axes=True,
                     label="raw atm transmission pwv=%.2f, elev=%i" % (pwv, elev))
            fig.step(atm_trans, "r", twin_axes=True,
                     label="pixel atm transmission")
            fig.legend(twin_axes=True, loc="lower right")
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
            t_arr = time_to_gps_ts(Time(
                    np.char.replace(tb_use["UTC"], "U", "T"), format="isot"))
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
                err_in = \
                    1 / (1 / err_use ** 2).proc_along_time("nansum").sqrt() / \
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
                zobs_ts * np.sqrt(stack + 1), write_header=os.path.join(
                        write_dir, data_file_header), pix_flag_list=pix_flag_list,
                plot=plot, plot_rms=plot_flux, plot_ts=False,
                reg_interest=reg_interest, plot_psd=plot_ts,
                plot_specgram=False, plot_show=plot_show, plot_save=plot_save)
        if table_save:
            beams_rms.to_table(orientation=ORIENTATION).write(os.path.join(
                    write_dir, "%s_rms.csv" % data_file_header), overwrite=True)
        flat_use = flat_flux.proc_along_time("nanmean") if \
            isinstance(flat_flux, type(beams_rms)) else flat_flux
        beams_sensitivity = 2 * np.sqrt(3) * beams_rms / abs(flat_use) / \
                            np.sqrt(zobs_ts.len_ * (stack + 1))
        if plot:
            fig = FigSpec.plot_spec(
                    zobs_flux, yerr=zobs_err, pix_flag_list=pix_flag_list,
                    color="k", label="spectrum")
            fig.step(beams_sensitivity, lw=.5, c="b", pix_flag_list=pix_flag_list,
                     label="predicted error")
            fig.imshow_flag(pix_flag_list=pix_flag_list)
            array_map = ObsArray(zobs_flux).array_map_
            fig.plot_all_spat(
                    [array_map.array_spec_llim_, array_map.array_spec_ulim_],
                    [0, 0], "k:")
            if plot_atm and trans_flag:
                fig.plot(atm_trans_raw, "c:", twin_axes=True,
                         label="raw atm transmission pwv=%.2f, elev=%i" %
                               (pwv, elev))
                fig.step(atm_trans, "r", twin_axes=True,
                         label="pixel atm transmission")
                fig.legend(twin_axes=True, loc="lower right")
            fig.legend()
            fig.set_title("%s spectrum" % data_file_header)
            if plot_show:
                plt.show()
            if plot_save:
                fig.savefig(os.path.join(
                        write_dir, "%s_spec.png" % data_file_header))
            plt.close(fig)
        if table_save:
            beams_sensitivity.to_table(orientation=ORIENTATION).write(
                    os.path.join(write_dir, "%s_predicted_err.csv" %
                                 data_file_header), overwrite=True)

    result = (zobs_flux, zobs_err, zobs_wt)
    if return_ts:
        result += (zobs_ts,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


# TODO: return intermediate result


def reduce_calibration(data_header, data_dir=None, write_dir=None,
                       write_suffix="", array_map=None, obs_log=None,
                       is_flat=False, pix_flag_list=None, flat_flux=1, flat_err=0,
                       cross=False, parallel=False, do_desnake=False,
                       ref_pix=None, do_smooth=False, do_ica=False,
                       spat_excl=None, do_clean=False, return_ts=False,
                       return_pix_flag_list=True, table_save=True, plot=True,
                       plot_ts=True, reg_interest=None, plot_flux=True,
                       plot_show=False, plot_save=True, analyze=False):
    """
    reduce data for general calibration that does not involve nodding or raster,
    but just continuous chop observations like pointing or focus
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_clean, do_ica),
                            ("desnake", "smooth", "clean" "ica")):
        if flag and method not in write_suffix:
            write_suffix += "_" + method
    result = reduce_beams(
            data_header=data_header, data_dir=data_dir, write_dir=write_dir,
            write_suffix=write_suffix, array_map=array_map, obs_log=obs_log,
            is_flat=is_flat, pix_flag_list=pix_flag_list, flat_flux=flat_flux,
            flat_err=flat_err, cross=cross, parallel=parallel,
            do_desnake=do_desnake, ref_pix=ref_pix, do_smooth=do_smooth,
            do_ica=do_ica, spat_excl=spat_excl, do_clean=do_clean,
            return_ts=return_ts | analyze,
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
            t_arr = time_to_gps_ts(Time(
                    np.char.replace(tb_use["UTC"], "U", "T"), format="isot"))
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
                 array_map=None, obs_log=None, is_flat=False, pix_flag_list=None,
                 flat_flux=1, flat_err=0, parallel=False, do_desnake=False,
                 ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                 do_clean=False,
                 return_ts=False, return_pix_flag_list=True, table_save=True,
                 plot=True, plot_ts=True, reg_interest=None, plot_flux=True,
                 plot_show=False, plot_save=True, analyze=False,
                 nod=False, use_hk=True, zpold_shape=ZPOLD_SHAPE):
    """
    plot raster of zpold

    :param bool nod: bool, flag whether the zpold is nodding, if True, it means
        there should be
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
    if (write_suffix != "") and (write_suffix[0] != "_"):
        write_suffix = "_" + write_suffix
    for flag, method in zip((do_desnake, do_smooth, do_clean, do_ica),
                            ("desnake", "smooth", "clean", "ica")):
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
                spat_excl=spat_excl, do_clean=do_clean, return_ts=True,
                return_pix_flag_list=True,
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
                spat_excl=spat_excl, do_clean=do_clean, return_ts=True,
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
                t_arr = time_to_gps_ts(Time(
                        np.char.replace(tb_use["UTC"], "U", "T"), format="isot"))
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
            raster=raster_flux,
            raster_wt=(1 / (beams_err ** 2).proc_along_time(method="nanmean").
                       sqrt()).data_[..., None],
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
    result += (stacked_result,)
    if return_pix_flag_list:
        result += (pix_flag_list,)

    return result


def reduce_zpoldbig(data_header, data_dir=None, write_dir=None, write_suffix="",
                    array_map=None, obs_log=None, is_flat=False,
                    pix_flag_list=None,
                    flat_flux=1, flat_err=0, parallel=False, do_desnake=False,
                    ref_pix=None, do_smooth=False, do_ica=False, spat_excl=None,
                    do_clean=False,
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
            spat_excl=spat_excl, do_clean=do_clean, return_ts=return_ts,
            return_pix_flag_list=return_pix_flag_list, table_save=table_save,
            plot=plot, plot_ts=plot_ts, reg_interest=reg_interest,
            plot_flux=plot_flux, plot_show=plot_show, plot_save=plot_save,
            analyze=analyze, nod=nod, use_hk=use_hk, zpold_shape=zpoldbig_shape)


def eval_performance(data_header, data_dir=None, write_dir=None, write_suffix="",
                     array_map=None, obs_log=None, pix_flag_list=None,
                     parallel=False, return_ts=False, table_save=True,
                     plot=True, plot_ts=True, reg_interest=None, plot_psd=True,
                     plot_specgram=True, plot_flux=True, plot_show=False,
                     plot_save=True):
    """
    Read a batch of beams and run analyze_performance on the time series. Be
    cautious that plot_specgram can be very slow
    """

    data_dir = str(data_dir) if data_dir is not None else os.getcwd()
    write_dir = str(write_dir) if write_dir is not None else os.getcwd()
    pix_flag_list = list(pix_flag_list).copy() if pix_flag_list is not None else []
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
