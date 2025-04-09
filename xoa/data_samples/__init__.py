#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available data files that are used by :mod:`xoa` examples.

.. autodata:: DATA_SAMPLES

"""
import os
import glob

_THIS_DIR = os.path.dirname(__file__)

#: Dictionary that contains the basename and full path of data sample files used by xoa examples
DATA_SAMPLES = dict(
    (os.path.basename(path), path)
    for path in glob.glob(os.path.join(_THIS_DIR, "*"))
    if not path.endswith(".py")
)


def get_data_sample(filename=None):
    """Get the absolute path to a sample file

    Parameters
    ----------
    filename: str, None
        Name of the sample. If ommited, a list of available samples
        name is returned.

    Returns
    -------
    str OR list(str)

    Example
    -------
    .. .ipython:: python

        @suppress
        from xoa import get_data_sample
        get_data_sample("croco.south-africa.surf.nc")
        get_data_sample()

    See also
    --------
    show_data_samples
    """
    if filename is None:
        return list(DATA_SAMPLES.keys())
    if filename not in DATA_SAMPLES:
        from ..__init__ import XoaError

        raise XoaError(
            f"Invalid data sample: '{filename}'.\n"
            + "Please use one of: "
            + ", ".join(DATA_SAMPLES)
        )
    return DATA_SAMPLES[filename]


def open_data_sample(filename, **kwargs):
    """Open a data sample with :func:`xarray.open_dataset` or :func:`pandas.read_csv`

    Parameters
    ----------
    filename: str
        File name of the sample.
        If not an existing sample, a warning is raised and if the path exists and has
        a csv or nc extension, it is opened.

    Returns
    -------
    xarray.Dataset, pandas.DataFrame

    Example
    -------
    .. .ipython:: python

        @suppress
        from xoa import open_data_sample
        open_data_sample("croco.south-africa.nc")


    See also
    --------
    get_data_sample
    show_data_samples
    """
    from ..__init__ import XoaError, xoa_warn

    try:
        path = get_data_sample(filename)
    except XoaError as e:
        msg = str(e) + ".\nThis function is for opening xoa internal sample file."
        path = filename
        if not os.path.exists(path):
            raise XoaError("Invalid path: " + path)
        else:
            msg += "\nTrying to open it..."
        xoa_warn(msg)
    if path.endswith("nc"):
        import xarray as xr

        return xr.open_dataset(path, **kwargs)
    if path.endswith("csv"):
        import pandas as pd

        return pd.read_csv(path, **kwargs)
    raise XoaError("Don't know haw to open this file")


def show_data_samples(full_paths=False):
    """Print the list of data samples

    Parameters
    ----------
    full_paths: bool
        Show full absolute paths.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import show_data_samples
        show_data_samples()

    See also
    --------
    get_data_samples
    """
    paths = get_data_sample()
    if full_paths:
        paths = list(DATA_SAMPLES.values())
    else:
        paths = list(DATA_SAMPLES)
    print(' '.join(paths))
