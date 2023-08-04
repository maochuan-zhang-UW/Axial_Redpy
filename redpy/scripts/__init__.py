"""
Module containing command line scripts.

The primary function of this module is to support allowing the user to
interact with REDPy's functionality from the command line. Scripts may also
be called in the interpreter, though with slightly different syntax (e.g.,
redpy.force_plot() instead of redpy-force-plot). Be aware that these scripts
almost all open and close a table, and if the same configuration file is
already in use it will complain of a resource lock.
"""
from redpy.scripts.backfill import backfill
from redpy.scripts.catfill import catfill
from redpy.scripts.clearjunk import clear_junk
from redpy.scripts.comparecatalog import compare_catalog
from redpy.scripts.createpdffamily import create_pdf_family
from redpy.scripts.createpdfoverview import create_pdf_overview
from redpy.scripts.createreport import create_report
from redpy.scripts.distantfamilies import distant_families
from redpy.scripts.extendtable import extend_table
from redpy.scripts.forceplot import force_plot
from redpy.scripts.initialize import initialize
from redpy.scripts.makemeta import make_meta
from redpy.scripts.plotjunk import plot_junk
from redpy.scripts.removefamily import remove_family
from redpy.scripts.removefamilygui import remove_family_gui
from redpy.scripts.removesmallfamily import remove_small_family
from redpy.scripts.writefamilylocations import write_family_locations
