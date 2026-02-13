"""Meta-data and conventions management"""

import os
import fnmatch

from platformdirs import user_config_dir

from .. import exceptions
from .. import misc
from . import configs
from . import general


_THISDIR = os.path.dirname(__file__)

# Joint variables and coords config specification file
INI_FILE = os.path.join(_THISDIR, "meta.ini")

#: User Meta config file
USER_META_FILE = os.path.join(user_config_dir("xoa"), "meta.cfg")

# Cache dict
_META_CACHE = {}


def get_cache():
    """Get the meta cache dict"""
    if not len(_META_CACHE):
        _META_CACHE.update(
            current=None,  # current active specs
            default=None,  # default xoa specs
            loaded_dicts={},  # for pure caching of dicts by key
            registered=[],  # for registration and matching purpose
        )
    return _META_CACHE


def get_matching_item_specs(da, loc="any"):
    """Get the item Meta specs that match this data array

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    dict or None

    See also
    --------
    MetaSpecs.match
    """
    meta_specs = get_meta_specs(da)
    cat, name = meta_specs.match(da, loc=loc)
    if cat:
        return meta_specs[cat][name]


def _same_attr_(da0, da1, attr):
    return (
        attr in da0.attrs
        and attr in da1.attrs
        and da0.attrs[attr].lower() == da1.attrs[attr].lower()
    )


def are_similar(da0, da1):
    """Check if two DataArrays are similar

    Verifications are performed in the following order:

    - ``standard_name`` attribute,
    - Matching MetaSpecs item name.
    - ``name`` attribute.
    - ``long_name`` attribute.

    Parameters
    ----------
    da0: xarray.DataArray
    da1: xarray.DataArray

    Return
    ------
    bool
    """
    # Standard name
    if _same_attr_(da0, da1, "standard_name"):
        return True

    # Cf name
    meta0 = get_matching_item_specs(da0)
    meta1 = get_matching_item_specs(da1)
    if meta0 and meta1 and meta0.name == meta1.name:
        return True

    # Name
    if da0.name and da0.name and da0.name == da1.name:
        return True

    # Long name
    return _same_attr_(da0, da1, "long_name")


def search_similar(obj, da):
    """Search in ds for a similar DataArray

    See :func:`is_similar` for what means "similar".

    Parameters
    ----------
    obj: xarray.Dataset, xarray.DataArray
        Dataset that must be scanned.
    da: xarray.DataArray
        Array that must be compared to the content of ``ds``

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    is_similar
    get_matching_item_specs
    """
    targets = misc.list_xr_names(obj, dims=False)
    for name in targets:
        if are_similar(obj[name], da):
            return obj[name]


class set_meta_specs(object):
    """Set the current Meta specs

    Parameters
    ----------
    meta_source: MetaSpecs, str, list, dict
        Either a :class:`MetaSpecs` instance or the name of a registered one,
        or an argument to instantiante one.

    See also
    --------
    get_meta_specs
    register_meta_specs
    get_registered_meta_specs
    """

    def __init__(self, meta_source):
        if isinstance(meta_source, str):
            meta_specs = get_meta_specs_from_name(meta_source, errors="ignore")
            if meta_specs:
                meta_source = meta_specs
        if not isinstance(meta_source, general.MetaSpecs):
            meta_source = general.MetaSpecs(meta_source)
        self.meta_cache = get_cache()
        self.old_specs = self.meta_cache["current"]
        self.meta_cache["current"] = self.specs = meta_source

    def __enter__(self):
        return self.specs

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_specs is None:
            self.meta_cache["current"] = None
        else:
            self.meta_cache["current"] = self.old_specs


def reset_cache(memory=False, **kwargs):
    """Reset the in memory meta specs cache

    Parameters
    ----------
    memory: bool
        Remove the in-memory cache.

        .. warning:: This may lead to unpredicted behaviors.

    """

    if "disk" in kwargs:
        exceptions.xoa_warn("Disk cachng is no longer supported", category="deprecation")
    if memory:
        meta_cache = get_cache()
        meta_cache["loaded_dicts"].clear()
        meta_cache["current"] = None
        meta_cache["default"] = None
        meta_cache["registered"].clear()


def show_cache():
    """Show the meta specs cache file"""
    exceptions.xoa_warn("Disk cachng is no longer supported", category="deprecation")


def get_meta_config_file(name):
    """Get the path of a meta config file given its short name"""
    if name.endswith(".cfg"):
        name = name[:-4]
    if name not in configs.META_CONFIGS:

        raise exceptions.XoaMetaError(
            "fInvalid meta config name '{name}'.\n"
            + "Please use on of: "
            + ", ".join(configs.META_CONFIGS)
        )
    return configs.META_CONFIGS[name]


@misc.ERRORS.format_function_docstring
def get_meta_specs_from_name(name, errors="warn"):
    """Get a registered Meta specs instance from its name

    Parameters
    ----------
    name: str
    {errors}

    Return
    ------
    MetaSpecs or None
        Issue a warning if not found
    """

    # Registered specs
    meta_cache = get_cache()
    for meta_specs in meta_cache["registered"][::-1]:
        if meta_specs["register"]["name"] and meta_specs["register"]["name"] == name.lower():
            return meta_specs

    # Internal specs
    if name in configs.META_CONFIGS:
        meta_specs = general.MetaSpecs(configs.META_CONFIGS[name])
        register_meta_specs(meta_specs)
        return meta_specs

    # Not found
    errors = misc.ERRORS[errors]
    msg = f"Unknown registration name for Meta specs: {name}"
    if errors == "raise":
        raise exceptions.XoaMetaError(msg)
    elif errors == "warn":
        exceptions.xoa_warn(msg)


def get_meta_specs_encoding(ds):
    """Get the ``meta_specs`` encoding value

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset

    Return
    ------
    str or None

    See also
    --------
    get_meta_specs_from_encoding
    """
    if ds is not None and not isinstance(ds, str):
        for source in ds.encoding, ds.attrs:
            for attr, value in source.items():
                if attr.lower() == "meta_specs":
                    return value


def get_meta_specs_from_encoding(ds):
    """Get a registered Meta specs instance from the ``meta_specs`` encoding value

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset

    Return
    ------
    MetaSpecs or None

    See also
    --------
    get_meta_specs_encoding
    """
    if ds is not None and not isinstance(ds, str):
        name = get_meta_specs_encoding(ds)
        if name is not None:
            return get_meta_specs_from_name(name, errors="warn")


def get_default_meta_specs(**kwargs):
    """Get the default Meta specifications"""

    if "cache " in kwargs:
        exceptions.xoa_warn("Disk cachng is no longer supported", category="deprecation")
    meta_cache = get_cache()
    if meta_cache["default"] is not None:
        return meta_cache["default"]

    # Setup
    meta_specs = general.MetaSpecs()
    meta_cache["default"] = meta_specs
    if not is_registered_meta_specs(meta_specs):
        register_meta_specs(meta_specs)
    return meta_specs


def get_meta_specs(name=None, cache="rw"):
    """Get the current or a registered Meta specifications instance

    Parameters
    ----------
    name: str, "current", "default", None, xarray.Dataset, xarray.DataArray
        "default" means the default xoa specs.
        "current" is equivalent to None and means the currents specs,
        which defaults to the xoa defaults!
        Else registration name for these specs or a data array or dataset
        that can be used to get the registration name if it set in the
        :attr:`meta_specs` attribute or encoding.
        When set, ``cache`` is ignored.
        Raises a :class:`XoaError` is case of invalid name.
    cache: str, bool, None
        Cache default specs on disk with pickling for fast loading.
        If ``None``, it defaults to boolean option :xoaoption:`meta.cache`.
        Possible string values: ``"ignore"``, ``"rw"``, ``"read"``, ``"write"``.
        If ``True``, it is set to ``"rw"``.
        If ``False``, it is set to ``"ignore"``.

    Return
    ------
    MetaSpecs
        None is return if no specs are found

    Raise
    -----
    XoaError
        When ``name`` is provided as a string and is invalid.
    """
    # Explicit request
    if name is None:
        name = "current"
    if not isinstance(name, str) or name not in ("current", "default"):
        # Registered name
        if isinstance(name, str):
            return get_meta_specs_from_name(name, errors="raise")

        # Name as dataset or data array so we infer the specs
        return infer_meta_specs(name)

    # Not named => current or default specs
    if name == "current":
        meta_cache = get_cache()
        if meta_cache.get("current") is None:
            meta_cache["current"] = get_default_meta_specs()
        meta_specs = meta_cache["current"]
    else:
        meta_specs = get_default_meta_specs()

    return meta_specs


def register_meta_specs(*args, **kwargs):
    """Register :class:`MetaSpecs` in a bank optionally with a name"""
    # Named arguments
    args = list(args)
    for name, meta_specs in kwargs.items():
        if not isinstance(meta_specs, general.MetaSpecs):
            meta_specs = general.MetaSpecs(meta_specs)
        meta_specs.name = name
        args.append(meta_specs)

    # Update the cache
    for meta_specs in args:
        meta_cache = get_cache()
        if not isinstance(meta_specs, general.MetaSpecs):
            meta_specs = general.MetaSpecs(meta_specs)
        if meta_specs not in meta_cache["registered"]:
            if meta_specs.name:  # replace if same name. warn?
                for rmetas in meta_cache["registered"]:
                    if rmetas.name and rmetas.name == meta_specs.name:
                        meta_cache["registered"].remove(rmetas)
            meta_cache["registered"].append(meta_specs)


def get_registered_meta_specs(current=True, reverse=True, named=False):
    """Get the list of registered MetaSpecs

    Parameters
    ----------
    current: bool
        Also include the current specs if any, always at the last position
    reverse: bool
        Reverse the list
    named: bool
        Make sure the returned MetaSpecs have a valid registration name

    Return
    ------
    list

    See also
    --------
    register_meta_specs
    """
    meta_cache = get_cache()
    metal = meta_cache["registered"]
    if reverse:
        metal = metal[::-1]
    if current and meta_cache["current"] is not None:
        metal.append(meta_cache["current"])
    if named is False:
        metadef = get_default_meta_specs()
        if metadef not in metal:
            metal.append(metadef)
    else:
        metal = [c for c in metal if c.name]
    return metal


def is_registered_meta_specs(name):
    """Check if given meta specs set is registered

    Parameters
    ----------
    name: str, MetaSpecs

    Return
    ------
    bool
    """
    for meta_specs in get_registered_meta_specs():
        if (
            isinstance(name, str)
            and meta_specs["register"]["name"]
            and meta_specs["register"]["name"] == name
        ):
            return True
        if isinstance(name, general.MetaSpecs) and name is meta_specs:
            return True
    return False


def get_meta_specs_matching_score(ds, meta_specs):
    """Get the matching score between ds data_vars and coord names and a MetaSpecs instance names

    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
    meta_specs: MetaSpecs

    Return
    ------
    float
        A percentage of the number of identified data arrays vs
        the total number of data arrays
    """
    hit = 0
    total = 0
    for cat in "data_vars", "coords":
        metanames = [
            meta_specs[cat].get_name(name, specialize=True) for name in meta_specs[cat].names
        ]
        if not hasattr(ds, "data_vars"):  # DataArray
            dsnames = [ds.name] if ds.name else []
        else:
            dsnames = list(getattr(ds, cat).keys())
        dsnames = [meta_specs.sglocator.parse_attr("name", dsname)[0] for dsname in dsnames]
        total += len(dsnames)
        hit += len(set(dsnames).intersection(metanames))
    if total == 0:
        return 0
    return 100 * hit / total


def infer_meta_specs(ds, named=False, from_attrs=True, from_score=True):
    """Get the registered MetaSpecs that are best matching this dataset

    This accomplished with some heurestics.
    First, the :attr:`meta_specs` global attribute or encoding of the dataset is compared
    with the name of all registered datasets.
    Second, a score based on the number of data_vars and coord names
    that are both in the meta_specs and the dataset is computed by :func:`get_meta_specs_matching_score`
    for the registered instances.
    Finally, if no matching dataset is found, the current one is returned.


    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
    named: bool
        Make sure the candidate MetaSpecs have a name
    from_attrs: bool
        Scan attributes to infer specs
    from_score: bool
        Compute the matching score to infer specs

    Return
    ------
    MetaSpecs
        The matching meta specs or the current ones

    See also
    --------
    register_meta_specs
    get_registered_meta_specs
    get_meta_specs_matching_score
    get_meta_specs
    get_meta_specs
    get_meta_specs_from_name
    get_meta_specs_from_encoding
    """
    # By registration name first
    meta_specs = get_meta_specs_from_encoding(ds)
    if meta_specs:
        return meta_specs

    # Candidates
    candidates = get_registered_meta_specs(named=named)

    # By attributes
    if from_attrs:
        for attrs in (ds.attrs, ds.encoding):
            if attrs:
                for meta_specs in candidates:
                    for attr, pattern in meta_specs["register"]["attrs"].items():
                        if attr in attrs:
                            if isinstance(pattern, str):
                                pattern = [pattern]
                            for pat in pattern:
                                if fnmatch.fnmatch(str(attrs[attr]).lower(), pat.lower()):
                                    return meta_specs

    # By matching score
    if from_score:
        best_score = -1
        for meta_specs in candidates:
            score = get_meta_specs_matching_score(ds, meta_specs)
            if score != 0 and score > best_score:
                best_meta_specs = meta_specs
                best_score = score
        if best_score != -1:
            return best_meta_specs

    # Fallback to current specs
    meta_specs = get_meta_specs("current")
    if named and not meta_specs.name:
        return
    return meta_specs


def assign_meta_specs(ds, name=None, register=False, set_encoding=True):
    """Set the ``meta_specs`` encoding to ``name`` in all data vars and coords

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset
    name: None, str, MetaSpecs, xarray.DataArray, xarray.Dataset
        If a :class:`MetaSpecs`, it must have a registration name :

        .. code-block:: ini

            [register]
            name=registration_name

        If not provided, :func:`infer_meta_specs` is called to infer
        the best named registered specs.

    register: bool
        Register the specs if name is a named, unregistered :class:`MetaSpecs` instance.
    set_encoding: bool
        Set the "meta_specs" encoding to name.

    Return
    ------
    xarray.Dataset, xarray.DataArray

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.meta import assign_meta_specs
        @suppress
        import xarray as xr
        ds = xr.Dataset({'temp': ('lon', [5])}, coords={'lon': [6]})
        assign_meta_specs(ds, "mycroco");
        ds.encoding
        ds.temp.encoding
        ds.lon.encoding

    """
    # Name as a MetaSpecs instance
    if name is None:
        meta_specs = infer_meta_specs(ds, named=True)
        if meta_specs.name:
            name = meta_specs.name
        else:
            return ds
    elif hasattr(name, "coords"):  # from a dataset/dataarray
        name = get_meta_specs_encoding(ds)
        if name is None:
            return ds
    if not isinstance(name, str):
        if not name.name:
            exceptions.xoa_warn("MetaSpecs instance has no registration name")
            return ds
        if register and not is_registered_meta_specs(name):
            register_meta_specs(name)
        name = name.name

    # Set as encoding
    if set_encoding:

        targets = [ds] + [ds[name] for name in misc.list_xr_names(ds, dims=False)]
        for target in targets:
            target.encoding.update(meta_specs=name)

    return ds


def infer_coords(ds):
    """Infer which of the data arrays of a dataset are coordinates

    When coordinates are found, it makes sure they are registered in the dataset
    as coordindates.

    Parameters
    ----------
    ds: xarray.Dataset

    See also
    --------
    MetaSpecs.infer_coords
    """
    return get_meta_specs(ds).infer_coords(ds)


# infer_coords.__doc__ = MetaSpecs.infer_coords.__doc__


@misc.ERRORS.format_function_docstring
def get_variant(ds, variants, errors="ignore"):
    """Try to find a unique generic data array in a dataset

    Parameters
    ----------
    ds: xarraya.Dataset
    variants: str, list(str)
        A single or a list of meta names
    {errors}

    Returns
    -------
    xarray.DataArray, None
    """
    meta_specs = get_meta_specs(ds)
    return meta_specs.get(ds, variants, errors=errors)
