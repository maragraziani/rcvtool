import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
print("Using Tensorflow {}".format(tf.__version__))
import keras
print("Using Keras {}".format(keras.__version__))
from keras.engine import Layer
import keras.backend as K
import numpy as np
import sys

"""
Credits for the Gradient Reversal Layer
https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
"""
def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    hp_lambda= tf.Print(hp_lambda, [hp_lambda], "HP_LAMBDA")
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        grad = tf.Print(tf.negative(grad), [tf.negative(grad)], "grad_reversal")
        final_val = grad * hp_lambda
        final_val = tf.Print(final_val, [final_val], "final_val")
        return [final_val]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        #self.hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda, name='hp_lambda')
    def build(self, input_shape):
        self.trainable_weights = []
        return
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  'hp_lambda': K.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
Classifier Loss that does not count unlabeled samples
"""
def classifier_loss(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    weights = tf.reduce_sum(y_true, 1)
    #weights=tf.Print(tf.reduce_sum(weights), [tf.reduce_sum(weights)], "weights: " )
    zero= tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(weights, zero)
    indices=tf.where(where) #indices where the weights are zero
    indices = tf.reshape(indices, [-1])

    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)

    n1 = 16
    n2 = 0
    sliced_y_true = tf.reshape(sliced_y_true, [n1,-1])
    sliced_y_pred = tf.reshape(sliced_y_pred, [n1,-1])
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sliced_y_true,logits=sliced_y_pred)

    #multiplier = (n1+n2) / n1
    #ce = tf.Print(ce, [ce], "ce: ")
    #ce = tf.scalar_mul(multiplier, ce)
    #ce = tf.Print(ce, [ce], "multiplier * ce: ")
    all_ce = tf.concat([ce, tf.zeros([n2])], axis=0)
    #all_ce = tf.Print(all_ce, [all_ce], "all_ce: ")
    avg_ce = tf.reduce_mean(all_ce)
    #avg_ce = tf.Print(avg_ce, [avg_ce], "avg_ce: ")

    # we CE
    return all_ce

def acc(y_true, y_pred):
    return K.mean(K.round(y_pred)*y_true + (1.-K.round(y_pred))*(1.-y_true))
#my_ce = my_classifier_loss(y_true, y_pred)
#with tf.Session() as sess:
#    print(my_ce.eval())

"""
Definitions of Concept Adversarial Models:
- Multilayer Perceptron (caMLP)
- ShallowCNN (caSCNN)
- Inception V3 (caIV3)
"""

class CAMLP():
    """Concept Adversarial MLP"""
    def feature_extractor(self, inp):
            '''
            This function defines the structure of the feature extractor part.
            '''
            if self.channels<3:
                out = keras.layers.Flatten(input_shape=(self.input_shape,self.input_shape))(inp)
            else:
                out = keras.layers.Flatten(input_shape=(self.input_shape,self.input_shape, self.channels))(inp)
            counter = 0
            while counter<self.height:
                out = keras.layers.Dense(self.width, activation=keras.layers.Activation('relu'))(out)
                counter +=1
            self.domain_invariant_features = out
            return out

    def classifier(self, inp):
        '''
        This function defines the structure of the classifier part.
        '''
        out = keras.layers.Dense(self.classes, activation="softmax", name="classifier_output")(inp)
        return out

    def concept_regressor(self, inp):
        '''
        This function defines the structure of the concept regression part.
        '''
        out = keras.layers.Dense(1, activation="linear", name="regression_output")(inp)
        return out

    def compile(self, optimizer):
        '''
        This function compiles the model based on the given optimization method and its parameters.
        '''
        #self.model.compile(optimizer=optimizer,
        #                   loss= 'categorical_crossentropy', metrics=['accuracy'])

        self.model.compile(optimizer=optimizer,
                           loss=classifier_loss,
                           metrics=['categorical_accuracy']
                           #loss={'classifier_output': 'categorical_crossentropy'},# 'regression_output': 'mean_squared_error'},
                           #loss_weights={'classifier_output': 1.0},# 'regression_output': 0.},
                           #metrics=['accuracy']
                           )

    def _build(self, nonlinear_regression=False):
            '''
            This function builds the network based on the Feature Extractor, Classifier and Regression parts.
            NOTE: the gradient reversal should be on the REGRESSION layer!
            '''
            mlp = keras.models.Sequential()
            mlp.add(keras.layers.Flatten(input_shape=(self.input_shape,self.input_shape)))
            counter = 0
            while counter<self.height:
                mlp.add(keras.layers.Dense(self.width, activation=keras.layers.Activation('relu')))
                counter+=1
            loss_function = 'categorical_crossentropy'
            activation = 'softmax'
            if self.classes == 2:
                loss_function = 'binary_crossentropy'
                activation = 'sigmoid'
            feature_output = mlp.layers[-1].output

            self.grl_layer = GradientReversal(hp_lambda=0.)
            feature_output_grl = self.grl_layer(feature_output)
            classifier_output = keras.layers.Dense(self.classes, activation=keras.layers.Activation(activation), name='classifier')(feature_output)
            if nonlinear_regression:
                print("Nonlinear regression with 2 dense layers")
                regression_intermediate = keras.layers.Dense(512, activation = keras.layers.Activation('relu'))(feature_output_grl)
                regression_output = keras.layers.Dense(1, activation = keras.layers.Activation('linear'), name='concept_regressor')(regression_intermediate)
            else:
                regression_output = keras.layers.Dense(1, activation = keras.layers.Activation('linear'), name='concept_regressor')(feature_output_grl)

            #inp = keras.layers.Input(shape=(self.input_shape,self.input_shape), name="main_input")
            #feature_output = self.feature_extractor(inp)
            #classifier_output = self.classifier(feature_output)
            '''
            ###


            labeled_feature_output = keras.layers.Lambda(lambda x:
                                                           K.switch(K.learning_phase(),
                                                                      K.concatenate([x[:int(self.batch_size//2)], x[:int(self.batch_size//2)]], axis=0), x),
                                                            output_shape=lambda x: x[0:])(feature_output_grl)
            ###
            '''

            model = keras.models.Model(inputs=mlp.input, output=[classifier_output, regression_output])#, regression_output])

            return model


    def __init__(self, deep=2, wide=512, channels=3, features=1, grl = 'auto', optimizer='SGD', lr=1e-2, epochs=10,
                 batch_size=32, input_shape=28, classes=10, summary = False, model_plot=False, nonlinear_regression=False, **kwargs):
        self.learning_phase = K.variable(1)
        self.domain_invariant_features = None
        self.width, self.height, self.channels = wide, deep, channels
        self.input_shape =  input_shape #(channels, width, eight)
        self.classes = classes
        self.features = features
        self.batch_size = batch_size
        self.grl = 'auto'
        # Set reversal gradient value
        if grl is 'auto':
            self.grl_rate = 0.0
        else:
            self.grl_rate = grl
        self.summary = summary
        self.model_plot = model_plot
        # Build the model
        self.model = self._build(nonlinear_regression=nonlinear_regression)
        # Print and Save the model summary if requested.
        if self.summary:
            self.model.summary()
        if self.model_plot:
            plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def batch_generator(self, trainX, trainY=None, cm=None, batch_size=1, shuffle=True):
        '''
        This function generates batches for the training purposes.
        '''
        if shuffle:
            index = np.random.randint(0, len(trainX) - batch_size)
        else:
            index = np.arange(0, len(trainX), batch_size)
        while trainX.shape[0] > index + batch_size:
            batch_images = trainX[index : index + batch_size]
            #useless for now
            #batch_images = batch_images.reshape(batch_size, self.channels, self.width, self.height)
            batch_cm = cm[index : index + batch_size]
            if trainY is not None:
                batch_labels = trainY[index : index + batch_size]
                yield batch_images, batch_labels, batch_cm
            else:
                yield batch_images, batch_cm
            index += batch_size

    def train(self, trainX, trainDX, trainY=None, cm=None, Dcm=None, epochs=1, batch_size=1, verbose=True, save_model=None):
            '''
            This function trains the model using the input and target data, and saves the model if specified.
            '''
            all_metrics=[]
            # Re-compile model to adopt new learning rate and gradient reversal value.
            lr = 1e-2
            import math
            n_batches = int((len(trainX)) / batch_size)
            print('len(trainX): {}, batch_size: {}, n_batches: {}'.format(len(trainX), batch_size, n_batches))
            self.compile(keras.optimizers.SGD(lr,clipvalue=.5))
            for cnt in range(epochs):
                # Settings for learning rate.
                p = np.float(cnt) / epochs
                # Uncomment for lr scheduling
                #lr = 0.01 / (1. + 10 * p)**0.75
                #lr = 1e-2
                # Settings for reverse gradient magnitude (if it's set to be automatically calculated, otherwise set by user.)
                #if self.grl is 'auto':
                    # Uncomment to activate regression block
                    #self.grl_layer.l = 2. / (1. + np.exp(-10. * p)) - 1
                    #self.grl_layer.l = 0.
                #print('Setting learning rate and reverse gradient penalty.')
                #print('lr: {}, lambda: {}'.format(lr, self.grl_layer.l))

                # Prepare batch data for the model training.
                #Labeled = self.batch_generator(trainX, trainY, cm=cm, batch_size=batch_size *2// 2)
                #UNLabeled = self.batch_generator(trainDX, cm=Dcm, batch_size=batch_size*2 // 2)

                all_metrics=[]
                c=0
                start=0
                #import pdb; pdb.set_trace()
                while(c < n_batches):
                    batchX = trainX[start:start+batch_size]
                    batchY = trainY[start:start+batch_size]
                    batch_cm = cm[start:start+batch_size]
                    #import pdb; pdb.set_trace()
                    #metrics = self.model.train_on_batch(batchX, batchY)
                    metrics = self.model.train_on_batch({'main_input': batchX}, #combined_batchX
                                                        {'classifier_output':  batchY, # batch2Y,
                                                         'regression_output':batch_cm
                                                        }) #'regression_output':combined_batchY})'''
                    if(math.isnan(metrics[0])):
                        import pdb; pdb.set_trace()
                    all_metrics.append(metrics)

                    start = start + batch_size
                    c+=1
                if verbose:
                    print("Epoch {}/{}, loss: {}, accuracy: {}".format(cnt+1,epochs, metrics[0], metrics))
                    #print(all_metrics)
                    #print("Epoch {}/{}\n\t[Cumulative_loss: {:.4}, Classifier_loss: {:.4}, Regression_loss: {:.4}]".format(cnt+1, epochs, metrics[0], metrics[1], metrics[2]))
                    #print("-- classifier CE: {}, accuracy: {}".format(metrics[3],metrics[5]))
                    #print('-- regressor MSE: {}'.format(metrics[7]))

            #import pdb; pdb.set_trace()
            '''


                # Loop over each batch and train the model.
                for batchX, batchY, batch_cm in Labeled:
                    # Get the batch for unlabeled data. If the batches are finished, regenerate the batches agian.
                    try:
                        batchDX, batch_Dcm = next(UNLabeled)
                    except:
                        UNLabeled= self.batch_generator(trainDX, cm=Dcm, batch_size=batch_size *2// 2)

                    # Combine the labeled and unlabeled images along with the discriminative results.
                    combined_batchX = np.concatenate((batchX, batchDX))
                    #batch2Y = np.concatenate((batchY, np.zeros((len(batchY),10))))#batchY)) #np.zeros((len(batchY),10))))
                    zeros=np.zeros((16,10))
                    #print([batchY[0]])
                    batch2Y = np.concatenate((batchY, zeros ))
                    #import pdb; pdb.set_trace()
                    regression_labels = np.concatenate((batch_cm, batch_Dcm))
                    #print('Regression labels.len: {}'.format(len(regression_labels)))
                    # Train the model

                    metrics = self.model.train_on_batch({'main_input': batchX}, #combined_batchX
                                                        {'classifier_output':  batchY, # batch2Y,
                                                         'regression_output':regression_labels[:16]
                                                        }) #'regression_output':combined_batchY})
                    all_metrics.append(metrics)
                import pdb; pdb.set_trace()
                # Print the losses if asked for.
                if verbose:
                    print("Epoch {}/{}\n\t[Cumulative_loss: {:.4}, Classifier_loss: {:.4}, Regression_loss: {:.4}]".format(cnt+1, epochs, metrics[0], metrics[1], metrics[2]))
                    print("-- classifier CE: {}, accuracy: {}".format(metrics[3],metrics[5]))
                    print('-- regressor MSE: {}'.format(metrics[7]))
            '''
            # Save the model if asked for.
            if save_model is not None and isinstance(save_model, str):
                if save_model[-3:] is not ".h5":
                    save_model = ''.join((save_model, ".h5"))
                self.model.save(save_model)
            elif save_model is not None and not isinstance(save_model, str):
                raise TypeError("The input must be a filename for model settings in string format.")

    def evaluate(self, testX, testY=None, weight_loc=None, save_pred=None, verbose=False):
            '''
            This function evaluates the model, and generates the predicted classes.
            '''
            if weight_loc is not None:
                self.compile(keras.optimizers.SGD())
                self.model.load_weights(weight_loc)
            _, yhat_class = self.model.predict(testX, verbose=verbose)
            if save_pred is not None:
                np.save(save_pred, yhat_class)
            if testY is not None and len(testY) == 2:
                acc = self.model.evaluate(testX, testY, verbose=verbose)
                if verbose:
                    print("The classifier and regression metrics for evaluation are [{}, {}]".format(acc[0], acc[1]))
            elif testY is not None and len(testY) == 1:
                acc = self.model.evaluate(testX, [np.ones((testY.shape[0], 2)), testY], verbose=verbose)
                if verbose:
                    print("The classifier metric for evaluation is {}".format(acc[1]))
