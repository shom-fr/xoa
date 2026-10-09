"""
Weights files of the interpolation and regridding classes

A weights file is a netcdf file whose groups each hold the weights of one grid, so that
a single file can serve all the grids of a model, like the ones of a staggered grid.
The name of a group is made of the kind of weights, the method and a fingerprint of the
grids, so that weights are found automatically and can never be loaded on another grid.

.. note:: Groups are appended to the file and never modified, and the file is not
    protected against several processes writing to it at the same time.
"""

# Copyright 2020-2026 Shom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import xarray as xr

from . import exceptions

#: Version of the format of the weights files, which is stored in the file
FORMAT_VERSION = 1


def get_group_name(kind, method, fingerprint):
    """Get the name of the group of weights

    Parameters
    ----------
    kind: str
        Kind of weights like ``"regrid"`` or ``"interp"``
    method: str
        Method like ``"bilinear"``
    fingerprint: str
        Fingerprint of the grids, as returned by :func:`xoa.misc.get_array_fingerprint`

    Return
    ------
    str
    """
    return f"{kind}_{method}_{fingerprint}"


def list_groups(path):
    """List the groups of a weights file, which is empty if the file does not exist"""
    if not os.path.exists(path):
        return []
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        return sorted(nc.groups)


def has_group(path, group):
    """Tell if a weights file exists and has a group"""
    return group in list_groups(path)


def describe_groups(path):
    """Describe the groups of a weights file

    Parameters
    ----------
    path: str

    Return
    ------
    list(dict)
        One dictionary per group with a ``"group"`` key, and the attributes of the group:
        ``kind``, ``method``, ``fingerprint``, ``src_fingerprint``, ``dst_fingerprint``,
        ``n_src`` and ``n_dst``. The fingerprints are the ones of the objects that
        made the weights, and the ones of the grids that
        :func:`xoa.grid.get_fingerprint` returns.
    """
    if not os.path.exists(path):
        return []
    import netCDF4

    out = []
    with netCDF4.Dataset(path) as nc:
        for name in sorted(nc.groups):
            group = nc.groups[name]
            info = {"group": name}
            for attr in group.ncattrs():
                value = group.getncattr(attr)
                info[attr] = value.item() if hasattr(value, "item") else value
            out.append(info)
    return out


def find_groups(path, fingerprint=None, kind=None, method=None):
    """Find the groups of a weights file that match criteria

    Parameters
    ----------
    path: str
    fingerprint: None, str
        Keep the groups of the weights that involve a grid with this fingerprint,
        either as the source, as the destination, or as both of them.
        Get it with :func:`xoa.grid.get_fingerprint`, or from the
        ``fingerprint``, ``src_fingerprint`` and ``dst_fingerprint`` attributes of
        :class:`xoa.regrid.Regridder` and :class:`xoa.interp.Interpolator`.
    kind: None, str
        ``"regrid"`` or ``"interp"``
    method: None, str
        Method like ``"bilinear"``

    Return
    ------
    list(str)
        Names of the groups

    Example
    -------
    .. code-block:: python

        fingerprint = xoa.grid.get_fingerprint(ds)
        xoa.weights.find_groups("weights.nc", fingerprint, method="conservative")
    """
    fingerprints = ("fingerprint", "src_fingerprint", "dst_fingerprint")
    found = []
    for info in describe_groups(path):
        if fingerprint is not None and fingerprint not in [info.get(k) for k in fingerprints]:
            continue
        if kind is not None and info.get("kind") != kind:
            continue
        if method is not None and info.get("method") != method:
            continue
        found.append(info["group"])
    return found


def is_legacy(path):
    """Tell if a file holds the weights of a single grid at its root, without fingerprint

    This was the format of the files before the support of groups.
    """
    if not os.path.exists(path):
        return False
    with xr.open_dataset(path) as ds:
        return "valid_dst_mask" in ds and "xoa_weights_format" not in ds.attrs


def save_group(path, group, variables, attrs=None):
    """Append a group of weights to a file, which is created if needed

    Parameters
    ----------
    path: str
    group: str
        Name of the group, see :func:`get_group_name`
    variables: dict
        Variables of the :class:`xarray.Dataset` to write
    attrs: dict, None
        Attributes of the group

    Return
    ------
    bool
        False if the group already exists, in which case nothing is written since
        the name of a group is made from the fingerprint of the grids.
    """
    if not os.path.exists(path):
        xr.Dataset(attrs={"xoa_weights_format": FORMAT_VERSION}).to_netcdf(path, mode="w")
    elif has_group(path, group):
        return False
    ds = xr.Dataset(variables, attrs=attrs)
    encoding = {name: {"zlib": True, "complevel": 1} for name in ds.data_vars}
    ds.to_netcdf(path, mode="a", group=group, encoding=encoding)
    return True


def load_group(path, group=None):
    """Load a group of weights in memory

    Parameters
    ----------
    path: str
    group: None, str
        Name of the group, or ``None`` to load the root of a legacy file.

    Return
    ------
    xarray.Dataset
    """
    with xr.open_dataset(path, group=group) as ds:
        return ds.load()


def select_group(path, group, fingerprint, method, n_dst, n_src, what):
    """Load the weights of a given grid from a file, with all the checks

    Parameters
    ----------
    path: str
        Weights file
    group: str
        Expected name of the group, see :func:`get_group_name`
    fingerprint: str
        Expected fingerprint of the grids
    method: str
        Expected method
    n_dst, n_src: int
        Expected sizes of the destination and source grids
    what: str
        Short description of the grids for the error messages

    Return
    ------
    xarray.Dataset

    Raises
    ------
    ValueError
        When the file has no matching weights
    """
    if has_group(path, group):
        ds = load_group(path, group)
        if ds.attrs.get("fingerprint") != fingerprint:
            raise ValueError(
                f"The fingerprint of the weights of group '{group}' "
                f"in '{path}' does not match the grids."
            )
    elif is_legacy(path):
        exceptions.xoa_warn(
            f"The weights file '{path}' has the legacy format: it has no fingerprint "
            "to check the grids, which is saved when the weights are written again."
        )
        ds = load_group(path)
        if ds.attrs.get("method") != method:
            raise ValueError(
                f"The weights file is for the {ds.attrs.get('method')} method, not "
                f"for the {method} one."
            )
    else:
        groups = ", ".join(list_groups(path)) or "none"
        raise ValueError(
            f"No weights for {what} and the {method} method in '{path}'. "
            f"Available groups: {groups}."
        )
    saved_n_dst = int(ds.attrs.get("n_dst", -1))
    saved_n_src = int(ds.attrs.get("n_src", -1))
    if saved_n_dst != n_dst or saved_n_src != n_src:
        raise ValueError(
            f"Loaded weights grid ({saved_n_dst}, {saved_n_src}) doesn't match "
            f"current grid ({n_dst}, {n_src})."
        )
    return ds
