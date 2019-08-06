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

def get_color_measure(image, mask=None, type=None, verbose=False):
    if mask is not None:
        print "A mask was specified"
        print "This feature has not been implemented yet"
        return None
    if type is None:
        print "No type was given"
        return None
    if type=='colorfulness':
        return colorfulness(image)
    else:
        return colorness(image, type, threshold=0, verbose=verbose)

def get_all_color_measures(image, mask=None, verbose=False):
    all_types = ['colrfulness',
                 'red',
                 'orange',
                 'yellow',
                 'green',
                 'cyano',
                 'blue',
                 'purple',
                 'magenta', 'dissimilarity'
                 'black',
                 'white'
                ]
    cms={}
    for type in all_types:
        if verbose:  print type
        cms[type]=get_color_measure(image,mask,type=type)
    return cms

def get_texture_measure(image, mask=None, type=None, verbose=False):
    if mask is not None:
        print "A mask was specified"
        print "This feature has been implemented in iMIMIC paper"
        return None
    if type is None:
        print "No type was given"
        return None
    return haralick(image, type, verbose=verbose)

def get_all_texture_measures(image, mask=None, verbose=False):
    all_types = ['dissimilarity',
                 'contrast',
                 'correlation',
                 'homogeneity',
                 'ASM',
                 'energy'
                ]
    cms={}
    for type in all_types:
        if verbose:  print type
        cms[type]=get_texture_measure(image,mask,type=type)
    return cms
