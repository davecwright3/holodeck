#!/usr/bin/env python3

"""
This module is intended to provide simulation-based inference tools for holodeck.
"""

import numpy as np
from holodeck import utils

VERBOSE = True

FLOOR_STRAIN_SQUARED = 1e-40
FLOOR_ERR = 1.0


def get_library(spectra, nfreqs, test_frac=0.0, center_measure="median"):
    """Get the GWB from a number of realizations.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    nfreqs : int
        The number of frequencies to train on, starting with the lowest in the
        library
    test_frac : float, optional
        The fraction of the data to reserve at the beginning as a test set
    center_measure : str, optional
        The measure of center for the dataset that the GP will be trained on. Can be
        either "mean" or "median"

    Returns
    -------
    gp_freqs : numpy.array
        The frequencies corresponding to the GWB data
    xobs : numpy.array
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

    gwb_spectra = gwb_spectra[test_ind:, :nfreqs, :] ** 2
    xobs = xobs[test_ind:, :]

    # Find all the zeros and set them to be h_c = 1e-20
    low_ind = gwb_spectra < FLOOR_STRAIN_SQUARED
    gwb_spectra[low_ind] = FLOOR_STRAIN_SQUARED

    # Get realizations that are all low. We will later use this
    # boolean array to set a noise floor
    # I've done it this way in case only certain frequencies have
    # all ~0 realizations.
    low_real = np.all(low_ind, axis=-1)

    # Find std
    # Where low_real is True, return 1.0
    # else return the std along the realization dimension
    err = np.where(low_real, FLOOR_ERR, np.std(np.log10(gwb_spectra), axis=-1))

    # these input output parameters create that output that is frequency dependent, it should find those frequency deps.
    return (gwb_spectra, theta)
