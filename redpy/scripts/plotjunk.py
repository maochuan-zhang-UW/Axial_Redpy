"""
Output the contents of the junk table for troubleshooting.

usage: redpy-plot-junk [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""
import argparse

from redpy.detector import Detector


def plot_junk(configfile='settings.cfg', verbose=False):
    """
    Output the contents of the junk table for troubleshooting.

    Creates images in a folder "junk" as well as a flat catalog
    "catalog_junk.txt" both in the main outputs directory for the run.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = Detector(configfile, verbose, opened=True)
    detector.output('junk')
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    plot_junk(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-plot-junk',
        description=('Output the contents of the junk table for '
                     'troubleshooting.'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
