"""
Module for handling functions related to actually creating outputs.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import redpy.outputs.html
import redpy.outputs.image
import redpy.outputs.mapping
import redpy.outputs.printing
import redpy.outputs.report
import redpy.outputs.timeline
