# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
r"""
Make "meta.html" to hold multiple meta overview pages.

Run this script to generate a file "meta.html" in a specified directory and
with a list of runs. This page gathers the 'meta_recent.html' tabbed
overviews within the output directories into a single page.

usage: redpy-make-meta [-h] [-v] [-r RUNS] [-p PATH] [-t TOPATH]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -r RUNS, --runs RUNS  comma separated list of runs to include, matching
                        the "groupname" in their configuration files
  -p PATH, --path PATH  relative path to where "meta.html" should be
                        created; defaults to "./runs"
  -t TOPATH, --topath TOPATH
                        relative path from "meta.html" to the runs; defaults
                        to same path
"""
import argparse
import os


def make_meta(runs='', path='./runs', topath='.', verbose=False):
    """
    Make "meta.html" to hold multiple meta overview pages.

    This page gathers the 'meta_recent.html' tabbed overviews within the
    output directories into a single page. This is intended to be used
    to monitor several runs simultaneously.

    Parameters
    ----------
    runs : str, optional
        Comma separated list of runs to include, matching the "groupname"
        in their configuration files.
    path : str, optional
        Relative path to where "meta.html" should be created; defaults to
        './runs' which is the default location for new run outputs.
    topath : str, optional
        Relative path from "meta.html" to the runs; defaults to same
        path.
    verbose : bool, optional
        Increase written print statements.

    """
    redpy.outputs.html.make_meta(runs, path, topath, verbose)


def main():
    """Handle run from the command line."""
    args = parse()
    make_meta(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description='Make "meta.html" to hold multiple meta overview pages.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-r', '--runs',
                        help=('comma separated list of runs to include, '
                              'matching their "groupName" in their '
                              'configuration files'))
    parser.add_argument('-p', '--path', default='./runs',
                        help=('relative path to where "meta.html" should be '
                              'created; defaults to "./runs"'))
    parser.add_argument('-t', '--topath', default='.',
                        help=('relative path from "meta.html" to the runs; '
                              'defaults to same path'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
