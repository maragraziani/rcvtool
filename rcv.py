import cv2
import numpy as np
from color_analysis import *
from texture_analysis import *
"""
Functions:

[TEST] get_color_measure(image, mask=None, type=None, verbose=True)
[TEST] get_texture_measure(image, mask=None, type=None, verbose=True)
[TEST] get_all_color_measures(image, mask=None)
[TEST] get_all_texture_measures(image, mask=None)

linear_regression(acts, measures, type='linear', evaluation=False, verbose=True)

mse(labels, predictions)
rsquared(labels, predictions)

"""

def get_color_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print "A mask was specified"
        print "This feature has not been implemented yet"
        return None
    if mtype is None:
        print "No type was given"
        return None
    if mtype=='colorfulness':
        return colorfulness(image)
    else:
        return colorness(image, mtype, threshold=0, verbose=verbose)

def get_all_color_measures(image, mask=None, verbose=False):
    all_types = ['colorfulness',
                 'red',
                 'orange',
                 'yellow',
                 'green',
                 'cyano',
                 'blue',
                 'purple',
                 'magenta', 
                 'black',
                 'white'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print mtype
        cms[mtype]=get_color_measure(image,mask=mask,mtype=mtype)
    return cms

def get_texture_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print "A mask was specified"
        print "This feature has been implemented in iMIMIC paper"
        return None
    if mtype is None:
        print "No type was given"
        return None
    return haralick(image, mask=mask, mtype=mtype, verbose=verbose)

def get_all_texture_measures(image, mask=None, verbose=False):
    all_types = ['dissimilarity',
                 'contrast',
                 'correlation',
                 'homogeneity',
                 'ASM',
                 'energy'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print mtype
        cms[mtype]=get_texture_measure(image,mask=mask,mtype=mtype)
    return cms
