#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright or © or Copr. Shom/Ifremer/Actimar
#
# stephane.raynaud@shom.fr, charria@ifremer.fr, wilkins@actimar.fr
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

# Imports
import os
import sys
import re
import shutil
from glob import glob

# Setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

# Get some specs from __init__.py
re_var_match = re.compile(r"^(__(\w+)__) = .+").match
specs = {}
valid_keys = ["version", "author", "email", "description"]
with open(os.path.join(os.path.dirname(__file__), "xoa/__init__.py")) as f:
    for line in f:
        line = line[:-1].strip()
        m = re_var_match(line)
        if m and m.group(2) in valid_keys:
            exec(line)
            specs[m.group(2)] = locals()[m.group(1)]
specs["author_email"] = specs.pop("email")
specs["long_description"] = specs.pop("description")

# Infos
specs.update(
    name="xoa",
    description="Xarray-based Ocean Analysis library",
    license="CeCiLL",
    url="https://github.com/VACUMM/xoa",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: CeCiLL",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: POSIX",
        "Operating System :: UNIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=['xoa']
)


def configuration():

    # Initialize
    config = Configuration()

    # Set options
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        quiet=True,
        delegate_options_to_subpackages=True,
    )

    # Add bin scripts
    scripts = []
    for pat in ["xoa", "*.py"]:
        pat = os.path.join(os.path.dirname(__file__), "bin", pat)
        scripts.extend(glob(pat))
    config.add_scripts(*scripts)

    # Add data samples
    config.add_data_dir('xoa/_samples')
    config.add_data_files("xoa/cf.ini", "xoa/cf.cfg")

    # Add extensions
    os.environ['LDFLAGS'] = "-shared"
    config.add_extension("xoa._interp", ["src/interp.f90"])

    # Env vars
    if "READTHEDOCS" in os.environ:
        for name, suffix in[
                ("FC", "gfortran"),
                ("F90", "gfortran"),
                ("F77", "gfortran"),
                ("AR", "ar"),
                ("GCC", "gcc"),
                ("LD", "ld")]:
            os.environ[name] = os.path.join(
                sys.prefix, "bin", "x86_64-conda_cos6-linux-gnu-"+suffix)

    return config


def main():

    # Setup config file
    if not os.path.exists("setup.cfg"):
        shutil.copy("setup.cfg.default", "setup.cfg")

    # Lauch setup
    setup(configuration=configuration, **specs)

if __name__ == "__main__":

    main()
