import cv2
import numpy as np
import keras
import keras.backend as K
import statsmodels.api as sm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.color
import skimage.feature

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


def haralick(image, mask=None, mtype=None, verbose=False):
    """
    Compute Haralick's descriptors of the Grey-Level Co-occurences matrix
    """
    # we use standard parameters in the computation of the glcm
    glcm = skimage.feature.greycomatrix(
            skimage.img_as_ubyte(skimage.color.rgb2gray(image)),
            [1],
            [0],
            symmetric=True,
            normed=True)
    return skimage.feature.greycoprops(glcm, mtype)


def colorfulness(img):
    """Colorfulness metric by .. & ..
    """
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(img.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def colorness(image, color_name, threshold=0, verbose=False):
    """ Colorness as defined in submission to ICCV
        blue-ness = #blue pixels / # pixels

        Use threshold = 0 for quantization of hue ranges
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # show color histograms for validation
    if verbose:
        h,s,v = hsv_histograms(image)
        plt.figure()
        plt.plot(h)
        plt.figure()
        plt.plot(s)
        plt.figure()
        plt.plot(v)
    # quantization of hue ranges
    # if threshold not 0, color name is changed into hue window
    if threshold == 0:
        hue_min, hue_max = quantize_hue_ranges(image, color_name)
        if verbose:
            print('hue min, hue max: ', hue_min, hue_max)
    else:
        h_point = color_picker(color_name)
        hue_min = round_hue(h_point[0][0][0]-threshold)
        hue_max = round_hue(h_point[0][0][0]+threshold)
        if verbose:
            print('hue min, hue max: ', hue_min, hue_max)
    if (hue_min == hue_max == 0) or (hue_min == 0 and hue_max == 255):
        #it is either black or white
        if color_name=='black':
            low_c = np.array([0,
                              0,
                              0])
            upp_c = np.array([hue_max,
                              100,
                              100])
        if color_name=='white':
            low_c = np.array([0,
                              0,
                              190])
            upp_c = np.array([hue_max,
                              50,
                              255])
        if verbose:
            print('low_c', low_c, 'upp_c', upp_c)
        mask = cv2.inRange(image, low_c, upp_c)
    elif hue_min>hue_max:
        low_c = np.array([0,
                      50,
                      77])
        upp_c = np.array([hue_max,
                      255,
                      255])
        mask1 = cv2.inRange(image, low_c, upp_c)

        low_c = np.array([hue_min,
                      50,
                      77])
        upp_c = np.array([180,
                      255,
                      255])
        mask2 = cv2.inRange(image, low_c, upp_c)
        mask = cv2.bitwise_or(mask1, mask1, mask2)
    else:
        low_c = np.array([hue_min,
                          50,
                          77])
        upp_c = np.array([hue_max,
                          255,
                          255])
        if verbose:
            print('low_c', low_c, 'upp_c', upp_c)
        mask = cv2.inRange(image, low_c, upp_c)
    if verbose:
        print(mask)
    res = cv2.bitwise_and(image, image, mask = mask)
    if verbose:
        plt.figure()
        plt.imshow(mask, cmap='Greys')
        plt.colorbar()
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.figure()
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))
    x,y,z = image.shape
    if verbose:
        print(np.sum(mask==255)/(float(x)*float(y)))
    return float(np.sum(mask==255))/(float(x)*float(y))
"""
Functions called by colorness module
"""
def hsv_histograms(image):
    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([image], [2], None, [256], [0, 256])
    #print np.mean(image[:,:,0])
    #print np.min(image[:,:,0])
    #print np.max(image[:,:,0])
    return hist_hue, hist_sat, hist_val

def color_picker(color_name):
    """
    Converts a color name into HSV values
    """
    brg_colors={}
    brg_colors['red']= np.uint8([[[0,0,255 ]]])
    brg_colors['orange'] = np.uint8([[[0,128,255 ]]])
    brg_colors['yellow'] = np.uint8([[[0,255,255 ]]])
    brg_colors['green'] = np.uint8([[[0,255,0 ]]])
    brg_colors['cyano'] = np.uint8([[[255,255,0 ]]])
    brg_colors['blue'] = np.uint8([[[255,0,0]]])
    brg_colors['purple'] = np.uint8([[[255,0,128]]])
    brg_colors['magenta'] = np.uint8([[[255,0,255 ]]])
    brg_colors['white'] = np.uint8([[[255,255,255 ]]])
    brg_colors['black'] = np.uint8([[[0,0,0 ]]])
    rgb_color_code = brg_colors[color_name]
    return cv2.cvtColor(rgb_color_code,cv2.COLOR_BGR2HSV)

def round_hue(hue_val):
    hues = np.arange(0,180)
    if hue_val<180:
        hue_def = hues[hue_val]
    else:
        hue_def = hues[(hue_val)%179]
    return hue_def

def quantize_hue_ranges(image, color_name):
    """
    Quantization of HSV space as in ICCV submission
    """
    if color_name == 'red':
        hue_min = 165
        hue_max = 10
    elif color_name == 'orange':
        hue_min = 10
        hue_max = 25
    elif color_name == 'yellow':
        hue_min = 25
        hue_max = 40
    elif color_name == 'green':
        hue_min = 40
        hue_max = 75
    elif color_name == 'cyano':
        hue_min = 75
        hue_max = 100
    elif color_name == 'blue':
        hue_min = 100
        hue_max = 125
    elif color_name == 'purple':
        hue_min = 125
        hue_max = 145
    elif color_name == 'magenta':
        hue_min = 145
        hue_max = 165
    elif (color_name == 'white' or color_name == 'black'):
        hue_min = 0
        hue_max = 255
    return hue_min, hue_max


def get_color_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print("A mask was specified.")
        print("This feature has not been implemented yet.")
        return None
    if mtype is None:
        if verbose:
            print("No type was given")
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
        if verbose:  print(mtype)
        cms[mtype]=get_color_measure(image,mask=mask,mtype=mtype, verbose=verbose)
    return cms

def get_texture_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        #print("A mask was specified")
        #print("This feature has been implemented in iMIMIC paper")
        return get_masked_texture(image, mask,method=mtype)
    if mtype is None:
        if verbose:
            print("No type was given")
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
    if mask is not None:
        return get_all_masked_textures(image, mask)
    for mtype in all_types:
        if verbose:  print(mtype)
        cms[mtype]=get_texture_measure(image,mask=mask,mtype=mtype)
    return cms

import skimage.feature

def get_masked_texture(image, mask, method='contrast'):
    if len(image.shape)<3:
        gray_img=image
    else:
        gray_img=cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
    copy = mask.copy()
    if np.max(copy)>1.:
        copy/=255.0
    copy[mask==0] = -1
    isolated_grayscale_mask = gray_img*mask
    isolated_grayscale_mask[copy==-1]=256
    p_image = np.asarray(isolated_grayscale_mask, dtype=np.int)
    glcm = skimage.feature.greycomatrix(p_image, [5], [0], 257, symmetric=True, normed=True)
    glcm=glcm[:256,:256]
    return skimage.feature.greycoprops(glcm, method)[0,0]
def get_all_masked_textures(image, mask):
    texture_feats={}
    methods=['contrast', 'correlation', 'ASM', 'energy', 'dissimilarity', 'homogeneity']
    for m in methods:
        texture_feats[m]=get_masked_texture(image, mask, method=m)
    return texture_feats

def binarize_image(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    scaled_image = (image-min_intensity) / (max_intensity-min_intensity)
    mask = (scaled_image > 0.5)*1
    return mask

def get_region_measure(image, mask=None, mtype=None, verbose=False):
    try:
        len(mask)
    except:
        mask = binarize_image(image)
        return get_region_measure(image, mask=mask, mtype=mtype, verbose=False)
    measure_properties = skimage.measure.regionprops(mask)[0]
    return measure_properties[mtype]

def get_all_region_measures(image, mask=None, verbose=False):
    all_types=['area',
               'perimeter',
               'eccentricity',
               'orientation'
              ]
    cms={}
    for mtype in all_types:
        if verbose:    print(mtype)
        cms[mtype]=get_region_measure(image, mask=mask, mtype=mtype)
    return cms

def get_batch_activations(model, layer, batch, pooling=None, labels=None):
    """
    gets a keras model as input, a layer name and a batch of data
    and outputs the network activations
    """
    get_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(layer).output])
    feats = get_layer_output([batch])
    if pooling=='AVG':
        return np.mean(feats[0], axis=(1,2))
    return feats[0]

def get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold=''):
    print("Not supported yet. Use get_batch_activations.")
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

def cluster_data(inputs, mode='KMeans', n_clusters=1, random_state=1):
    if mode=='DBSCAN':
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=15, min_samples=30).fit(inputs)
    if mode=='KMeans':
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state).fit(inputs)
    if mode=='NearestN':
        from sklear.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_clusters, algorithm='ball_tree').fit(indices)
        return None #not finished yet

def local_linear_regression(inputs, y, max_clusters, random_state=1, verbose=False):
    # find clusters
    # solve regression in the clusters
    # returns a
    clustering = cluster_data(inputs, mode='KMeans', n_clusters=max_clusters, random_state=random_state)
    if verbose:
        print(clustering.labels_)
    return clustering, clustering.labels_

def get_rcv(acts, measures, type='global linear', max_clusters=1, evaluation=True, random_state=1, verbose=True):
    """"
    Returns the RCV
    """
    if type=='global linear':
        #rcv_result = linear_regression(acts, measures, random_state=random_state, verbose=True)
        if evaluation:
            from sklearn.model_selection import train_test_split
            train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                            measures,
                                                                            test_size=0.3,
                                                                            random_state=random_state)
            rcv_result = linear_regression(train_acts, train_meas, random_state=random_state, verbose=True)
            rsquared = rcv_result.rsquared
            test_data_ = sm.add_constant(test_acts)
            predictions = rcv_result.predict(test_data_)
            mse_test = compute_mse(test_meas, predictions)
            r2_test = compute_rsquared(test_meas, predictions)
        #rcv_result = linear_regression(acts, measures, random_state=random_state, verbose=True)
        if verbose:
            print("Global linear regression under euclidean assumption")
            print("Random state: ", random_state)
            print("R2: ", rcv_result.rsquared)
            #if evaluation:
            #    print("MSE: ", mse)
        print("TEST mse: {}, r2: {}".format(mse_test, r2_test))
        #print(rcv_result.summary())
        return rcv_result
    elif type=='local linear':
        if verbose:
            print("Local linear regression under Euclidean assumption")
        if evaluation:
            local_rcvs = {}
            from sklearn.model_selection import train_test_split
            train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                            measures,
                                                                            test_size=0.3,
                                                                            random_state=random_state)
            #import pdb; pdb.set_trace()
            clusterer, clustering_labels = local_linear_regression(train_acts, train_meas, max_clusters)
            n_clusters = np.max(clustering_labels) + 1 # we start from 0 label
            for cluster_id in range(n_clusters):
                datapoint_idxs = np.argwhere(clustering_labels==cluster_id).T[0]
                rcv_result = linear_regression(train_acts[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
                rsquared = rcv_result.rsquared
                local_rcvs[cluster_id] = rcv_result
                if verbose:
                    print("Cluster no. {}".format(cluster_id))
                    print("Random state: ", random_state)
                    print("R2: ", rcv_result.rsquared)
            #(n,d)=test_acts.shape
            #centroids=np.zeros((n*n_clusters, d))
            #test_acts_duplicates=np.zeros((n*n_clusters,d))
            #for cluster_id in range(n_clusters):
            #    centroids[cluster_id*n:cluster_id*n+n]=cluster_centers[cluster_id]
            #    test_acts_duplicates[cluster_id*n:cluster_id*n+n]=test_acts
            #import pdb; pdb.set_trace()
            #distance_from_centroids = np.sum(np.power(test_acts_duplicates - centroids)
            avg_mse = 0
            avg_r2 = 0
            close_clusters=clusterer.predict(test_acts)
            for cluster_id in range(n_clusters):
                data_idxs = np.argwhere(close_clusters==cluster_id).T[0]
                test_data_ = sm.add_constant(test_acts[data_idxs])
                predictions = local_rcvs[cluster_id].predict(test_data_)
                mse = compute_mse(test_meas[data_idxs], predictions)
                r2 = compute_rsquared(test_meas[data_idxs], predictions)
                if verbose:
                    print("TEST cluster id: {}, mse: {}, r2: {}".format(cluster_id, mse, r2))
                avg_mse+=mse
                avg_r2+=r2
            avg_r2/=n_clusters
            avg_mse/=n_clusters
            if verbose:
                print("Cumulative MSE: {}, Avg R2: {}".format(avg_mse, avg_r2))
            return train_acts, clustering_labels, avg_mse, avg_r2
    elif type =='local UMAP':
        if verbose:
            print("Local linear regression on UMAP clustering with euclidean distances (UMAP)")
        import umap
        local_rcvs = {}
        from sklearn.model_selection import train_test_split
        train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                        measures,
                                                                        test_size=0.3,
                                                                        random_state=random_state)
        transform = umap.UMAP(n_neighbors=15,#15
                     min_dist=0.3,
                     random_state=random_state,
                     metric='euclidean').fit(train_acts)
        train_embedding=transform.embedding_

        clusterer, clustering_labels = local_linear_regression(train_embedding, train_meas, max_clusters)

        n_clusters = np.max(clustering_labels) + 1 # we start from 0 label
        for cluster_id in range(n_clusters):
            datapoint_idxs = np.argwhere(clustering_labels==cluster_id).T[0]
            rcv_result = linear_regression(train_acts[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
            #rcv_result = linear_regression(train_embedding[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
            rsquared = rcv_result.rsquared
            local_rcvs[cluster_id] = rcv_result
            if verbose:
                print("Cluster no. {}".format(cluster_id))
                print("Random state: ", random_state)
                print("R2: ", rcv_result.rsquared)
            avg_mse = 0
            avg_r2 = 0

        test_embedding = transform.transform(test_acts)
        close_clusters=clusterer.predict(test_embedding)
        for cluster_id in range(n_clusters):
            data_idxs = np.argwhere(close_clusters==cluster_id).T[0]
            test_data_ = sm.add_constant(test_acts[data_idxs])
            #test_data_ = sm.add_constant(test_embedding[data_idxs])
            predictions = local_rcvs[cluster_id].predict(test_data_)
            mse = compute_mse(test_meas[data_idxs], predictions)
            r2 = compute_rsquared(test_meas[data_idxs], predictions)
            if verbose:
                print("TEST cluster id: {}, mse: {}, r2: {}".format(cluster_id, mse, r2))
            avg_mse+=mse
            avg_r2+=r2
        avg_r2/=n_clusters
        avg_mse/=n_clusters
        print("Cumulative MSE: {}, Avg R2: {}".format(avg_mse, avg_r2))

        return transform, clusterer, train_acts, clustering_labels, avg_mse, avg_r2, local_rcvs #, train_meas

    elif type=='global manifold':
        if verbose:
            print("Global linear regression on unknown manifold")
    return rcv_result
