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
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
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
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=[]
        embeddings=[]
        batch_size=self.batch_size
        # specifying what to save: note, this part changes from network to network
        layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0

        # training batch by batch and appendinh the outputs
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            print epoch_number
            batch_number = 0
            embedding_=[]
            for l in layers_of_interest:
                space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                embedding_.append(space)
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])

                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[2]
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3]

                history.append(outs[0])
                batch_number+=1

            np.save('{}training_emb_e{}'.format(lcp_gnf,epoch_number), embedding_)
            del embedding_
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings

    def train(self, dataset):
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0

        try:
            shape1, shape2 = y_train.shape()
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