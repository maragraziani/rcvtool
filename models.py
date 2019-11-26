import keras
import numpy as np
import keras.backend as K
import tensorflow as tf
import os

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
    ce = tf.Print(ce, [ce], "ce: ")
    #ce = tf.scalar_mul(multiplier, ce)
    #ce = tf.Print(ce, [ce], "multiplier * ce: ")
    all_ce = tf.concat([ce, tf.zeros([n2])], axis=0)
    all_ce = tf.Print(all_ce, [all_ce], "all_ce: ")
    avg_ce = tf.reduce_mean(all_ce)
    avg_ce = tf.Print(avg_ce, [avg_ce], "avg_ce: ")
    return ce

class MLP():
    '''
    Multilayer Perceptron for experiments
    - on MNIST in SDLCV2019
    - on the extension of the regression (ongoing)

    Params
    deep: int
      Default 2
    wide: int
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 28
    n_classes: int
      Default 10
    '''

    def __init__(self, deep=2, wide=512, optimizer='SGD', lr=1e-2, epochs=10,
                 batch_size=32, input_shape=28, n_classes=10, **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)

        mlp = keras.models.Sequential()
        mlp.add(keras.layers.Flatten(input_shape=(input_shape,input_shape)))
        counter = 0
        while counter<deep:
            mlp.add(keras.layers.Dense(wide, activation=keras.layers.Activation('relu')))
            counter+=1
        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        mlp.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=mlp.input, output=mlp.output)
        optimizer = keras.optimizers.SGD(lr=lr, clipvalue=.5)
        model.compile(optimizer=optimizer,
                      loss= classifier_loss, #loss_function,
                      metrics=['categorical_accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train_and_compute_rcvs(self, dataset, lcp_gnf='0.8lcp_'):
        '''
        Train and Compute RCVs
          Saves the embeddings at each epoch in a npy file.
        dataset: Object of class either MNISTRandom, ImageNet10Random or Cifar10Random
          gives the object with rhe training data (dataset.x_train, dataset.y_train)
        lcp_gnf, string
          says if the dataet is corrupted with label corruption (lcp)
          or gaussian noise in the inputs (gnf). Specify the respective lcp (x.xlcp_),
          or gnf values (x.xgnf_) followed by the name of the corruption, f.e.
          0.8lcp_ for label corruption with probability 0.8
          or 0.5gnf_ for gaussian noise fraction of 0.5
        '''

        x_train = dataset.x_train
        y_train = dataset.y_train
        # check if the y have a categorical distr
        try:
            shape1, shape2 = y_train.shape
        except:
            y_train = keras.utils.to_categorical(y_train)
        try:
            shape1, shape2 = dataset.y_test.shape
        except:
            dataset.y_test = keras.utils.to_categorical(y_test)
        history=[]
        embeddings=[]
        batch_size=self.batch_size
        # specifying what to save: note, this part changes from network to network
        layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        print(layers_of_interest)
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        print(self.model.metrics_tensors)
        epoch_number = 0

        # training batch by batch and appendinh the outputs
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            #print(epoch_number)
            batch_number = 0
            embedding_=[]
            for l in layers_of_interest:
                space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                embedding_.append(space)
            import pdb; pdb.set_trace()
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])

                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[1] #2
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[2] #3

                history.append(outs[0])
                batch_number+=1
            evaluation=self.model.evaluate(dataset.x_test, dataset.y_test, batch_size=self.batch_size)
            print('Epoch: {} Train Acc: {} Test Acc: {}'.format(epoch_number, outs[1], evaluation[1]))
            #if epoch_number % 10:
            np.save('{}training_emb_e{}'.format(lcp_gnf,epoch_number), embedding_)
            del embedding_
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings

    def train(self, dataset):
        x_train = dataset.x_train
        y_train = dataset.y_train
        #x_train = x_train / 255.0

        try:
            shape1, shape2 = y_train.shape
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)
        self.training_history=history

    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)

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

    def _train(self, trainX, trainDX, trainY=None, cm=None, Dcm=None, epochs=1, batch_size=1, verbose=True, save_model=None):
            '''
            This function trains the model using the input and target data, and saves the model if specified.
            '''
            for cnt in range(epochs):
                # Prepare batch data for the model training.
                Labeled = self.batch_generator(trainX, trainY, cm=cm, batch_size=batch_size // 2)
                UNLabeled = self.batch_generator(trainDX, cm=Dcm, batch_size=batch_size // 2)

                # Settings for learning rate.
                #p = np.float(cnt) / epochs
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

                # Re-compile model to adopt new learning rate and gradient reversal value.
                #self.compile(keras.optimizers.SGD(lr))

                # Loop over each batch and train the model.
                for batchX, batchY, batch_cm in Labeled:
                    # Get the batch for unlabeled data. If the batches are finished, regenerate the batches agian.
                    try:
                        batchDX, batch_Dcm = next(UNLabeled)
                    except:
                        UNLabeled= self.batch_generator(trainDX, cm=Dcm, batch_size=batch_size // 2)


                    # Combine the labeled and unlabeled images along with the discriminative results.
                    combined_batchX = np.concatenate((batchX, batchDX))
                    batch2Y = np.concatenate((batchY, np.zeros((len(batchY),10))))
                    #batch2Y = tf.concat((batchY, tf.zeros((len(batchY),10), dtype=tf.dtypes.float32)))
                    #batch2Y = np.concatenate((batchY, batchY)) #np.zeros((len(batchY),10))))
                    regression_labels = np.concatenate((batch_cm, batch_Dcm))
                    #print('Regression labels.len: {}'.format(len(regression_labels)))
                    # Train the model
                    print(combined_batchX.shape, batch2Y.shape)
                    metrics = self.model.train_on_batch(combined_batchX, batch2Y)

                # Print the losses if asked for.
                if True:
                    print("Epoch {}: {} ".format(cnt,metrics))

class shallowCNN():
    '''
    Convolutional Neural Network for experiments on ImageNet
    input, crop(2,2),
    conv(200,5,5), bn, relu, maxpool(3,3),
    conv(200,5,5), bn, relu, maxpool(3,3),
    dense(384), bn, relu,
    dense(192), bn, relu,
    dense(n_classes), softmax

    Params
    deep: int (how many convolution blocks)
      Default 2
    wide: int (how many neurons in the first dense connection)
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 32
    n_classes: int
      Default 10
    '''

    def __init__(self, deep=2, wide=384, optimizer='SGD', lr=1e-2, epochs=9,
                 batch_size=16, input_shape=299, n_classes=10, save_fold='', **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)
        if input_shape<227:
            cropping=0
        else:
            cropping = (input_shape-227)/2

        cnn = keras.models.Sequential()
        cnn.add(keras.layers.Cropping2D(cropping=((cropping,cropping),(cropping,cropping)), input_shape=(input_shape,input_shape,3)))
        counter = 0
        while counter<deep:
            cnn.add(keras.layers.Conv2D(200, (5,5)))
            cnn.add((keras.layers.BatchNormalization()))
            cnn.add(keras.layers.Activation('relu'))
            cnn.add(keras.layers.MaxPool2D(pool_size=(3,3)))
            counter+=1
        cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(wide))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.Dense(wide/2))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))

        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        cnn.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=cnn.input, output=cnn.output)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.save_fold = save_fold
        self.deep = deep
        self.wide=wide

    def train(self, dataset):

        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0
        x_train -= np.mean(x_train)
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train = y_train[idxs_train]

        x_test = dataset.x_test
        y_test = dataset.y_test
        x_test = x_test / 255.0
        x_test -= np.mean(x_test)
        idxs_test = np.arange(len(x_test))
        np.random.shuffle(idxs_test)
        x_test = np.asarray(x_test[idxs_test])
        y_test = y_test[idxs_test]


        try:
            shape1, shape2 = y_train.shape
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        try:
            shape1, shape2 = y_test.shape
        except:
            y_test = keras.utils.to_categorical(y_test, self.n_classes)
        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test))
        self.training_history=history

    def train_and_compute_rcvs(self, dataset):

        x_train = dataset.x_train/255.
        x_train -= np.mean(x_train)
        y_train = dataset.y_train
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train= np.asarray(y_train)
        y_train = y_train[idxs_train]

        try:
            shape1, shape2 = y_train.shape
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=[]
        embeddings=[]
        batch_size=self.batch_size
        print(self.model.summary())

        #layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        if self.deep==2:
            layer_idxs = [9,13,16]
        if self.deep==3:
             layer_idxs = [9,12,14]
        if self.deep==4:
             layer_idxs = [9,12,15,19,22]
        if self.deep==5:
             layer_idxs = [9,12,15,18,22,25]

        layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in layer_idxs]
        print('loi', layers_of_interest)
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0


        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            print(epoch_number)
            batch_number = 0
            embedding_=[]

            for l in layers_of_interest:
                #print 'in layer ', l
                #print 'output shape ', self.model.get_layer(l).output.shape
                #print 'metrics tensors, ', self.model.metrics_tensors
                if len(self.model.get_layer(l).output.shape)<=2:
                    space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                else:
                    x = self.model.get_layer(l).output.shape[-3]
                    y = self.model.get_layer(l).output.shape[-2]
                    z = self.model.get_layer(l).output.shape[-1]
                    space = np.zeros((len(x_train), x*y*z))

                embedding_.append(space)
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])
                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]= outs[2].reshape((min(batch_size,len(outs[2])),-1))
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3].reshape((len(outs[3]),-1))
                embedding_[2][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[4].reshape((len(outs[4]),-1))
                #embedding_[3][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[5].reshape((len(outs[5]),-1))

                history.append(outs[0])
                batch_number+=1
            #print self.save_fold
            source = self.save_fold#'/mnt/nas2/results/IntermediateResults/Mara/probes/imagenet/2H_lcp0.5'
            c=0
            if True:
                for l in layers_of_interest:
                    if 'max_pooling' in l:

                        tosave_= embedding_[c] # np.mean(embedding_[c].reshape(11800200, 23*23,200), axis=1)
                        np.save('{}/shallowcnn_training_emb_e{}_l{}'.format(source,epoch_number, l), tosave_)
                    else:
                        np.save('{}/shallowcnn_training_emb_e{}_l{}'.format(source,epoch_number, l), embedding_[c])
                    c+=1
            del embedding_
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings

    def _custom_eval(self, x, y, batch_size):
        ## correcting shape-related issues
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        y = y.reshape(y.shape[0],-1)
        #
        scores = []
        val_batch_no = 0
        start_batch = val_batch_no
        end_batch = start_batch + batch_size
        tot_batches = len(y) / batch_size
        # looping over data
        while val_batch_no < tot_batches:
            score = self.model.test_on_batch(x[start_batch:end_batch, :299, :299, :3],
                                             y[start_batch:end_batch])
            scores.append(score[1])
            val_batch_no += 1
            start_batch = end_batch
            end_batch += batch_size
        #print("Val: {}".format(np.mean(np.asarray(scores))))
        return np.mean(np.asarray(scores))

    def train_and_monitor_with_rcvs(self, dataset, layers_of_interest=[], directory_save='',custom_epochs=0):
        '''
        Train and Monitor with RCVs
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
          Similar to train and compute RCVs, we just keep track
          of accuracy, partial accuracy (split in true and false labels)
          and we keep track of the embeddings corresponding to true and
          false labels.
          The function saves the embeddings at each epoch in a npy file.
          The mask of corrupted labels is saved in a separated npy file.
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        Inputs:
        dataset: Object of class either MNISTRandom, ImageNet10Random or Cifar10Random
          gives the object with rhe training data (dataset.x_train, dataset.y_train)
        name, string
          says the dataset name and if the dataet is corrupted with label corruption (lcp)
          or gaussian noise in the inputs (gnf). For example, if the datset is imagenet and we want to specify
          the respective lcp (x.xlcp_), or gnf values (x.xgnf_)
          we write datasetname_x.x followed by the name of the corruption, f.e.
          imagenet_0.8lcp_ for label corruption with probability 0.8
          or imagenet_0.5gnf_ for gaussian noise fraction of 0.5
        layers_of_interest, list
          allows to specify which layers we want to extract the embeddings from.
          ex.[6,11,14]
        '''
        directory_save = self.save_fold
        # train data with the original orderng (not shuffled yet)
        x_train = dataset.x_train/255.
        x_train -= np.mean(x_train)
        y_train = dataset.y_train
        # validation data with original orderng (not shuffled yet)
        x_val = np.asarray(dataset.x_test, dtype=np.float64)
        x_val -= np.mean(x_val)
        y_val = dataset.y_test
        # setting the seed for random
        try:
            np.random.seed(dataset.seed)
        except:
            np.random.seed(0)
        # mask of bool values set to true if the corresponding datapoint
        # was corrupted
        train_mask = dataset.train_mask
        # We shuffle the dataset indeces in a new array
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        # List of corrupted and uncorrupted indeces in
        # the original ordering of the data
        corrupted_idxs = np.argwhere(train_mask == True)
        uncorrupted_idxs = np.argwhere(train_mask == False)
        try:

            np.save('{}/corrupted_idxs.npy'.format(directory_save), corrupted_idxs)
        except:
            print("ERROR saving corr idxs")
        try:
            np.save('{}/uncorrupted_idxs.npy'.format(directory_save), uncorrupted_idxs)
        except:
            print("ERROR saving uncorr idxs")
        ## x_train and y_train contain the data with the new shuffling
        orig_x_train=x_train
        orig_y_train=y_train
        x_train = np.asarray(x_train[idxs_train])
        y_train = y_train[idxs_train]
        #y_train = dataset.y_train
        #if x_train
        #x_train = x_train / 255.0
        # converting the labels to categorical
        try:
            shape1, shape2 = y_train.shape
        except:
            y_train = keras.utils.to_categorical(y_train)
        # variables for logs and monitoring
        history=[]
        embeddings=[]
        batch_size=self.batch_size
        print(self.model.summary())
        ##### NOTE: the loi change from model to model
        layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in layers_of_interest]
        print('loi', layers_of_interest)
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            #print(epoch_number)
            batch_number = 0
            embedding_=[]
            for l in layers_of_interest:
                #print 'in layer ', l
                #print 'output shape ', self.model.get_layer(l).output.shape
                #print 'metrics tensors, ', self.model.metrics_tensors
                if len(self.model.get_layer(l).output.shape)<=2:
                    space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                else:
                    x = self.model.get_layer(l).output.shape[-3]
                    y = self.model.get_layer(l).output.shape[-2]
                    z = self.model.get_layer(l).output.shape[-1]
                    space = np.zeros((len(x_train), x*y*z))

                embedding_.append(space)
            while batch_number <= n_batches:
                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])
                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[2].reshape((min(batch_size,len(outs[2])),-1))
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3].reshape((len(outs[3]),-1))
                embedding_[2][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[4].reshape((len(outs[4]),-1))
                #embedding_[3][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[5].reshape((len(outs[5]),-1))
                #print outs, outs
                history.append(outs[0])
                batch_number+=1
            c=0
            for l in layers_of_interest:
                np.save('{}_training_emb_e{}_l{}'.format(name,epoch_number, l), embedding_[c])
                c+=1
            del embedding_
            # here we check the partial accuracy
            if epoch_number %10 == 0:
                corrupted_acc = self._custom_eval(orig_x_train[corrupted_idxs],
                                                  orig_y_train[corrupted_idxs],
                                                  batch_size
                                                 )
                uncorrupted_acc = self._custom_eval(orig_x_train[uncorrupted_idxs],
                                                  orig_y_train[uncorrupted_idxs],
                                                  batch_size
                                                 )
                try:
                    with open(directory_save+'/corr_acc.txt', 'a') as log_file:
                        log_file.write("{}, ".format(corrupted_acc))
                except:
                        log_file = open(directory_save+'/corr_acc.txt', 'w')
                        log_file.write("{}, ".format(corrupted_acc))

                try:
                    with open(directory_save+'/uncorr_acc.txt', 'a') as log_file:
                        log_file.write("{}, ".format(uncorrupted_acc))
                except:
                    log_file = open(directory_save+'/uncorr_acc.txt', 'w')
                    log_file.write("{}, ".format(uncorrupted_acc))

            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings


    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)
