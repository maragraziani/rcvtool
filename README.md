# rcv-tool

# Repository with the main tools for computing Regression Concept Vectors

Options:

- compute concept measures of color and texture on the whole image
- compute concept measures of color, texture and shape on a masked region

functions:

get_color_measure(image, mask=None, type=None, verbose=True)
get_texture_measure(image, mask=None, type=None, verbose=True)
get_all_color_measures(image, mask=None)
get_all_texture_measures(image, mask=None)

- compute linear regression
- compute ridge regression
- compute local linear regression

function:

linear_regression(acts, measures, type='linear', evaluation=False, verbose=True)

- evaluate regression on training or held-out data, with rsquared and mse
- evaluate adjusted rsquared on training and held-out data
- evaluate angle between to rcvs

functions:

mse(labels, predictions)
rsquared(labels, predictions)
