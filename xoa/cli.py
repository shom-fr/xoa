#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commandline interface for xoa
"""
# Copyright or © or Copr. Shom, 2020
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

import argparse

from .__init__ import (
    show_info, show_versions, show_paths, show_options)


def get_parser(formatter_class=argparse.ArgumentDefaultsHelpFormatter):

    parser = argparse.ArgumentParser(
        description="xoa interface",
        formatter_class=formatter_class)
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_info = subparsers.add_parser('info', help='info about xoa')
    parser_info.add_argument('category', help='info category', nargs='?',
                             choices=('all', 'paths', 'versions', 'options'),
                             default='all')
    parser_info.set_defaults(func=main_info)

    return parser


def main(argv=None, formatter_class=argparse.ArgumentDefaultsHelpFormatter):

    parser = get_parser(formatter_class=formatter_class)
    args = parser.parse_args(argv)
    args.func(parser, args)


def main_info(parser, args):
    if args.category == "all":
        show_info()
    elif args.category == "versions":
        show_versions()
    elif args.category == "paths":
        show_paths()
    elif args.category == "options":
        show_options()
