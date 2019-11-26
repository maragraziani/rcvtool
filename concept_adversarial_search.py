import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.__version__
import tensorflow as tf
from adconcept import *
import sys
import os
import numpy as np

#specify folder to save and create logs
# script arguments
# 1. saving folder
# 2. epochs for only classifier training
# 3. epochs of adversarial training
# 4. epochs of only classifier second round
# 5. alpha for the regression
# 6. alpha for the classifier
folder = 'cadvexps/{}/'.format(sys.argv[1])
os.mkdir(folder)

#instantiate model and save plot
K.clear_session()
damlp = CAMLP()
print(damlp.model.summary())
keras.utils.vis_utils.plot_model(damlp.model, to_file='{}model_plot.png'.format(folder))
# data load
trainX = np.load('./MNIST/cadv/data/trainX.npy') #mnist.x_train / 255.
testX = np.load('./MNIST/cadv/data/testX.npy')
trainY = np.load('./MNIST/cadv/data/trainY.npy') #keras.utils.to_categorical(mnist.y_train)
testY = np.load('./MNIST/cadv/data/testY.npy')
cm = np.load('./MNIST/cadv/data/orientations.npy')

tot_epochs = int(sys.argv[2])
alpha_regression = float(sys.argv[5])
alpha_class = float(sys.argv[6])
initial_lamb= float(sys.argv[7])
print("Alpha regression: {}".format(alpha_regression))
print("Only classifier")
for e in range(tot_epochs):
    p = np.float(e) / tot_epochs
    lr = 0.01 / (1. + 10 * p)**0.75
    damlp.model.compile(keras.optimizers.SGD(lr,clipvalue=.5),
                    loss = [classifier_loss, 'mean_squared_error'],
                    loss_weights=[alpha_class, alpha_regression],
                    metrics=['categorical_accuracy'])
    print("lr: {}, lambda: {}, regression_weight: {}".format(K.eval(damlp.model.optimizer.lr), K.eval(damlp.grl_layer.hp_lambda), lr_regression))
    damlp.model.fit(trainX, [trainY, np.asarray(cm)], batch_size=16, epochs=1)
last_epoch = tot_epochs

if int(sys.argv[3])>0:
    tot_epochs = int(sys.argv[3]) + last_epoch

    print("Adversarial")
    for e in range(last_epoch, tot_epochs):
        p = np.float(e) / tot_epochs
        lr = 0.01 / ((1. + 10 * p)**0.75)

        damlp.grl_layer.l = 2. / (1. + np.exp(-1. * p)) - 1
        lmb = 2. / (1. + np.exp(-1. * p)) - 1
        K.set_value(damlp.grl_layer.hp_lambda, lmb)
        lr_regression = alpha_regression
        damlp.model.compile(keras.optimizers.SGD(lr,clipvalue=.5),
                        loss = [classifier_loss, 'mean_squared_error'],
                        loss_weights=[alpha_class, lr_regression],
                        metrics=['categorical_accuracy'])
        print("lr: {}, lambda: {}, regression_weight: {}".format(K.eval(damlp.model.optimizer.lr), lmb, lr_regression))
        #print("Lr check: {}".format(K.eval(damlp.model.optimizer.lr)))
        damlp.model.fit(trainX, [trainY, np.asarray(cm)], batch_size=16, epochs=1)

    last_epoch = tot_epochs
    tot_epochs = int(sys.argv[4]) + last_epoch
    print("Only classifier round II")
    for e in range(tot_epochs):
        p = np.float(e) / tot_epochs
        lr = 0.01 / (1. + 10 * p)**0.75
        K.set_value(damlp.grl_layer.hp_lambda, 0.)
        damlp.model.compile(keras.optimizers.SGD(lr,clipvalue=.5),
                        loss = [classifier_loss, 'mean_squared_error'],
                        loss_weights=[1., 0.0],
                        metrics=['categorical_accuracy'])
        damlp.model.fit(trainX, [trainY, np.asarray(cm)], batch_size=16, epochs=1)

print("Eval: ", damlp.model.evaluate(testX, [testY, np.asarray(cm[:10000])], batch_size=16))
damlp.model.save_weights('{}model_weights.h5'.format(folder))
y_pred = damlp.model.predict(testX)[0]
np.save('{}predictions.h5'.format(folder), y_pred)
import keras.backend as K
l= 'dense_2'
inputs = trainX
get_layer_output = K.function([damlp.model.layers[0].input],
                              [damlp.model.get_layer(l).output])
feats = get_layer_output([inputs])[0]
feats.shape
np.save('{}adversarial_dense_2_acts.npy'.format(folder), feats)
