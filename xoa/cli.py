#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commandline interface for xoa
"""
# Copyright 2020-2021 Shom
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

import argparse

from .__init__ import (
    show_info, show_versions, show_paths, show_options)


def get_parser(formatter_class=argparse.ArgumentDefaultsHelpFormatter):

    parser = argparse.ArgumentParser(
        description="xoa interface",
        formatter_class=formatter_class)
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_info = subparsers.add_parser(
        'info', help='info about xoa')
    parser_info.add_argument(
        'category', help='info category', nargs='?',
        choices=('all', 'paths', 'versions', 'options'), default='all')
    parser_info.set_defaults(func=main_info)

    parser_reset_cf_cache = subparsers.add_parser(
        'reset_cf_cache', help='remove the CF cache file')
    parser_reset_cf_cache.set_defaults(func=main_reset_cf_cache)

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


def main_reset_cf_cache(parser, args):
    from . import cf
    cf.reset_cache(disk=True)
    print("Removed CF cache file: "+cf.USER_CF_CACHE_FILE)
