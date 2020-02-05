#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools
"""
# Copyright or © or Copr. Shom/Ifremer/Actimar
#
# stephane.raynaud@actimarshom.fr, charria@ifremer.fr, wilkins@actimar.fr
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import re
from collections import UserString


class SGLocator(object):
    """Staggered grid location parsing and formatting utility

    Parameters
    ----------
    name_format: str
        A string containing the string patterns ``{root}`` and ``{loc}}``,
        which defaults to ``"{root}_{loc}"``
    """

    formats = {'name': '{root}_{loc}',
               'standard_name': '{root}_at_{loc}_location',
               'long_name': '{root} at {loc} location'}

    re_match = {
        'standard_name': re.compile(
            formats['standard_name'].format(
                root=r'(\w+)', loc=r'([a-zA-Z])'), re.I).match,
        'long_name': re.compile(
            formats['long_name'].format(
                root=r'([\w ]+)', loc=r'([a-zA-Z])'), re.I).match
        }

    def __init__(self, name_format=None):

        # Init
        self.formats = self.formats.copy()
        self.re_match = self.re_match.copy()
        self._name_format = name_format

        # The name case
        if name_format:
            self.formats['name'] = name_format
        self.re_match['name'] = re.compile(
            self.formats['name'].format(root=r'(\w+)', loc=r'([a-zA-Z])'),
            re.I).match

    def match(self, obj, attr, root, loc=None):
        """Check if an attribute is matching a location

        Parameters
        ----------
        obj: object, xarray.DataArray
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        root: str
        loc: None, letter, False
            - None: any
            - letters: one of these locations
            - False: no location

        Return
        ------
        bool or loc
        """
        if not hasattr(obj, attr):
            return False
        value = getattr(obj, attr).lower()
        root = root.lower()
        if loc is False:
            return value == root
        m = self.re_match[attr](value)
        if m is None:
            return False
        if m.group(1) == root:
            if loc is None:
                return m.group(2)
            for lc in loc:
                if lc == m.group(2):
                    return lc
        return False

    def format_attr(self, attr, root, loc):
        """Format a attribute at a specified location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        root: str
        loc: False, letter

        Example
        -------
        >>> sg = SGLocator()
        >>> sg.format_attr('standard_name', 'sea_water_temperature', 't')
        'sea_water_temperature_at_t_location'
        >>> sg.format_attr('standard_name', 'sea_water_temperature', False)
        'sea_water_temperature'

        Return
        ------
        str
        """

        if not loc:
            return root
        if attr == 'long_name':
            loc = loc.upper()
        return self.formats[attr].format(root=root, loc=loc)
