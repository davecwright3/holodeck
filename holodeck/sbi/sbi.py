#!/usr/bin/env python3

"""
This module is intended to provide simulation-based inference tools for holodeck.
"""

import warnings

import kalepy as kale
import numpy as np
from ceffyl.bw import bandwidths as bw
from holodeck import utils
from KDEpy import FFTKDE

VERBOSE = True

FLOOR_STRAIN = 1e-20


def get_library(spectra, nfreqs, test_frac=0.0):
    """Get the GWB and parameters from a number of realizations.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    nfreqs : int
        The number of frequencies to train on, starting with the lowest in the
        library
    test_frac : float, optional
        The fraction of the data to reserve at the beginning as a test set

    Returns
    -------
    gwb_spectra : numpy.array
        The filtered GWB data
    theta : numpy.array
        A numpy array containing the parameters used to generate each GWB in `spectra`

    Examples
    --------
    FIXME: Add docs.

    """
    # Filter out NaN values which signify a failed sample point
    # shape: (samples, freqs, realizations)
    gwb_spectra = spectra["gwb"]
    xobs = spectra["sample_params"]
    bads = np.any(np.isnan(gwb_spectra), axis=(1, 2))

    if VERBOSE:
        utils.my_print(
            f"Found {utils.frac_str(bads)} samples with NaN entries.  Removing them from library."
        )
    # when sample points fail, all parameters are set to zero.  Make sure this is consistent
    if not np.all(xobs[bads] == 0.0):
        raise RuntimeError(
            f"NaN elements of `gwb` did not correspond to zero elements of `sample_params`!"
        )
    # Select valid spectra, and sample parameters
    gwb_spectra = gwb_spectra[~bads]
    theta = xobs[~bads]
    # Make sure old/deprecated parameters are not in library
    if "mmb_amp" in spectra.attrs["param_names"]:
        raise RuntimeError(
            "Parameter `mmb_amp` should not be here!  Needs to be log-spaced (`mmb_amp_log10`)!"
        )

    # Cut out portion for test set later
    test_ind = int(gwb_spectra.shape[0] * test_frac)

    if VERBOSE:
        utils.my_print(
            f"setting aside {test_frac} of samples ({test_ind}) for testing, and choosing {nfreqs} frequencies"
        )

    gwb_spectra = gwb_spectra[test_ind:, :nfreqs, :]
    xobs = xobs[test_ind:, :]

    # Find all the zeros and set them to be h_c = 1e-20
    low_ind = gwb_spectra < FLOOR_STRAIN
    gwb_spectra[low_ind] = FLOOR_STRAIN
    # get the frequency
    freqs = spectra["fobs"][:nfreqs]

    # these input output parameters create that output that is frequency dependent,
    # it should find those frequency deps.
    return (freqs, gwb_spectra, theta)


def hc_to_log10rho(hc, freqs, Tspan):
    hcsq = hc**2
    # turn predicted hcsq to psd/T to log10_rho
    psd = hcsq/(12*np.pi**2 *
                 freqs**3 * Tspan)
    log10_rho = 0.5*np.log10(psd)

    return log10_rho

def find_nearest_row(theta_grid, theta):
    """Given a array of shape (N,) or (1,N) , find the nearest (euclidean) row in a grid (M,N).

    Parameters
    ----------
    theta_grid : array_like
        The grid to search over.
    theta : array_like
        The array to calculate distance from.

    Returns
    -------
    ind_close : int
        The row index in the grid that is closest to the input array.

    Examples
    --------
    import numpy as np
    >>> x = np.arange(20).reshape(5,4)
    >>> x
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> y = np.array([4,6,6,7])
    >>> x[find_nearest_row(x,y)]
    array([4, 5, 6, 7])

    """
    # Take the difference. This should broadcast along the rows
    diff = theta_grid - theta

    # This takes the sum of squares over columns. It gives a euclidean distance for each row in the grid.
    distsq = np.einsum("ij,ij->i", diff, diff)

    # Find the index that matches the closest squared distance
    ind_close = np.argmin(distsq)

    return ind_close


def density(data, bw, kernel='epanechnikov', kde_func='FFTKDE',
            rho_grid=np.linspace(-15.5, 0, 10000),
            take_log=True, reflect=True, supress_warnings=True,
            return_kde=False, kde_kwargs={}):
    """
    Calculate Kernel Density Estimation (KDE) for an MCMC data chain.

    Parameters
    ----------
    data : array-like
        MCMC chain to calculate bandwidths.
    bw : float or str
        Bandwidth of KDE. This can be a number or a string accepted by the
        chosen KDE function.
    kernel : str, optional
        Name of the kernel to be used for the KDE.
    kde_func : str, optional
        KDE function to be used, chosen from ['kalepy', 'FFTKDE'].
    rho_grid : array-like, optional
        Grid of log10rho values to calculate probability density functions.
    take_log : bool, optional
        Return log probability density.
    reflect : bool, optional
        Include reflecting boundaries.
    suppress_warnings : bool, optional
        Suppress warnings from taking the log of 0.
    return_kde : bool, optional
        Return the initialized KDE function if True.
    kde_kwargs : dict, optional
        Other keyword arguments for the KDE function.

    Returns
    -------
    density : array
        Array of (log) probability density values.
    kde : KDE object, optional
        Initialized KDE function if return_kde is True.
    """
    if supress_warnings:  # supress warnings from taking log of zero
        warnings.filterwarnings('ignore')

    # if rho_grid is smaller than data range, cut off data to avoid error
    data = data[data > rho_grid.min()]
    data = data[data < rho_grid.max()]

    # initialise kalepy if chosen and fit data
    if kde_func == 'kalepy':
        kde = kale.KDE(data, bandwidth=bw,
                       kernel=kernel, **kde_kwargs)

        if reflect:
            lo_bound = rho_grid.min()
        else:
            lo_bound = None

        density = kde.density(rho_grid, probability=True,
                              reflect=[lo_bound, None])[1]

    # initialise KDEpy.FFTKDE if chosen and fit data
    elif kde_func == 'FFTKDE':
        if kernel == 'epanechnikov':  # change name for FFTKDE
            kernel = 'epa'

        kde = FFTKDE(bw=bw, kernel=kernel, **kde_kwargs)

        # reflect lower boundary
        if reflect:
            lo_bound = rho_grid.min()
            data = np.concatenate((data,
                                   2 * lo_bound - data))
            data = data[data >= lo_bound]
        else:
            data = data

        kde = kde.fit(data)  # fit data
        density = kde.evaluate(rho_grid)

    if take_log:  # switch to take log pdf
        density = np.log(density)

    if return_kde:
        return (density, kde)

    else:
        return density


def make_kdes_for_library(library, nfreqs=30, grid=np.linspace(-15.5,-1,10000), save=False):
    freqs, gwb, theta = get_library(library, nfreqs)

    # broadcast
    log10rho = hc_to_log10rho(gwb, freqs[None, :, None], 1 / freqs[0])

    # allocate array
    log10rho_kdes = np.full((*log10rho.shape[:-1],grid.size),np.nan,np.float64)
    for i in range(log10rho.shape[0]):
        for j in range(log10rho.shape[1]):
            try:
               log10rho_kdes[i,j,:] = density(log10rho[i,j,:],bw.sj(log10rho[i,j,:]), rho_grid=grid)
            except ValueError:
                break

    # filter out the samples that had any nan realizations
    bads = np.any(np.isnan(log10rho_kdes), axis=(1,2))
    log10rho_kdes = log10rho_kdes[~bads]
    theta = theta[~bads]
    if save:
        np.savez("holodeck-kdes", theta=theta, log10rho_grid=grid, log10rho_kdes=log10rho_kdes)
    return theta, log10rho_kdes
