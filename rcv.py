import cv2
import numpy as np
from color_analysis import *
from texture_analysis import *
import keras
import keras.backend as K
import statsmodels.api as sm

"""
Functions:

[TEST] get_color_measure(image, mask=None, type=None, verbose=True)
[TEST] get_texture_measure(image, mask=None, type=None, verbose=True)
[TEST] get_all_color_measures(image, mask=None)
[TEST] get_all_texture_measures(image, mask=None)

get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold='')
[NOT WORKING ATM]

get_rcv(acts, measures, type='linear', evaluation=False, verbose=True)

predict_with_rcv maybe? 

compute_mse(labels, predictions)
compute_rsquared(labels, predictions)

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

def get_batch_activations(model, layer, batch, labels=None):
    """
    gets a keras model as input, a layer name and a batch of data
    and outputs the network activations
    """
    get_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(layer).output])
    feats = get_layer_output([batch])
    return feats[0]

def get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold=''):
    print "todo"
    return None

"""Support function for get_rcv"""
def linear_regression(inputs, y, random_state=1, verbose=False):
    inputs = sm.add_constant(inputs)
    model = sm.OLS(y,inputs)
    results = model.fit()
    return results

def compute_mse(labels, predictions):
    errors = labels - predictions
    sum_squared_errors = np.sum(np.asarray([pow(errors[i],2) for i in range(len(errors))]))
    mse = sum_squared_errors / len(labels)
    return mse

def compute_rsquared(labels, predictions):
    errors = labels - predictions
    sum_squared_errors = np.sum(np.asarray([pow(errors[i],2) for i in range(len(errors))]))
    # total sum of squares, TTS
    average_y = np.mean(labels)
    total_errors = labels - average_y
    total_sum_squares = np.sum(np.asarray([pow(total_errors[i],2) for i in range(len(total_errors))]))
    #rsquared is 1-RSS/TTS
    rss_over_tts =   sum_squared_errors/total_sum_squares
    rsquared = 1-rss_over_tts
    return rsquared
"""end of support functions"""

def cluster_data(inputs, mode='DBSCAN'):
    if mode=='DBSCAN':
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=15, min_samples=30).fit(inputs)

def local_linear_regression(inputs, y, random_state=1, verbose=False):
    # find clusters 
    # solve regression in the clusters
    # returns a     
    clustering = cluster_data(inputs, mode='DBSCAN')
    print clustering.labels_
    return clustering.labels_
    
def get_rcv(acts, measures, type='global linear', evaluation=False, random_state=1, verbose=True):
    """" 
    Returns the RCV
    """
    if type=='global linear':
        rcv_result = linear_regression(acts, measures, random_state=random_state, verbose=True)
        if evaluation:
            rsquared = rcv_result.rsquared
            train_data_ = sm.add_constant(acts)
            predictions = rcv_result.predict(train_data_)
            mse = compute_mse(measures, predictions)
        if verbose:
            print "Global linear regression under euclidean assumption"
            print "Random state: ", random_state
            print "R2: ", rcv_result.rsquared
            try:
                print "MSE: ", mse
            except:
                pass
            print rcv_result.summary()
    elif type=='local linear':
        if verbose:
            print "Local linear regression under Euclidean assumption"
            local_linear_regression(acts, measures)
    elif type=='global manifold':
        if verbose:
            print "Global linear regression on unknown manifold"
    return
       
        
