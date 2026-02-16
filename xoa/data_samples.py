#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available data files that are used by :mod:`xoa` examples.

.. autodata:: DATA_SAMPLES

"""
import os

_THIS_DIR = os.path.dirname(__file__)


import os
import pooch

pooch.get_logger().setLevel("DEBUG")

POOCH = pooch.create(
    path=pooch.os_cache("xoa"),
    base_url="https://github.com/shom-fr/data-samples/raw/refs/heads/main/OCEANO/",
    # version_dev="main",
    env="SHOM_DATA_SAMPLES",
)

#: Registry of data samples
REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "data_samples.txt")

POOCH.load_registry(REGISTRY_FILE)


def get_data_sample(sample_name=None):
    """Fetch sample data file

    Downloads sample file from repository if needed and returns local path.

    Parameters
    ----------
    sample_name : str
        Sample file name (e.g., "MODELS/CROCO/SOUTH-AFRICAN/croco.south-africa.surf.nc").

    Returns
    -------
    str
        Absolute path to cached sample file.
    """
    if sample_name is None:
        file_names = []
        with open(REGISTRY_FILE) as f:
            for line in f:
                if len(line) > 1 and not line.startswith("#"):
                    file_names.append(line.split()[0])
        return file_names
    return POOCH.fetch(sample_name)


#: DEPRECATED! Dictionary that contains the basename and full path of data sample files used by xoa examples
DATA_SAMPLES = {}


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
        open_data_sample("MODELS/CROCO/SOUTH-AFRICAN/croco.south-africa.surf.nc")


    See also
    --------
    get_data_sample
    show_data_samples
    """
    from xoa import XoaError, xoa_warn

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
    raise XoaError("Don't know how to open this file")


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
        paths = [os.path.join(POOCH.abspath, path) for path in paths]
    print(' '.join(paths))
