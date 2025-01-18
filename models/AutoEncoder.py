import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
import datetime

class AutoEncoder():
    def __init__(self, 
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_strides,
                 z_dim,
                 use_batch_norm = False,
                 use_dropout = False,
                 epoch = 0
    ):
        self.name = 'autoencoder'
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.epoch = epoch
        
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        
        self._build()
 

    def _build(self):
        ### THE ENCODER
        encoder_input = tf.keras.layers.Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
    
        for i in range(self.n_layers_encoder):
            x = tf.keras.layers.Conv2D(
                filters = self.encoder_conv_filters[i],
                kernel_size = self.encoder_conv_kernel_size[i],
                strides = self.encoder_conv_strides[i],
                padding  = 'same',
                name = 'encoder_conv_' + str(i)
            )(x)
            x = tf.keras.layers.LeakyReLU()(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.use_dropout:
                x = tf.keras.layers.Dropout(rate = 0.25)(x)
              
        shape_before_flattening = tf.keras.backend.int_shape(x)[1:] # shape for 1 data
        
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(self.z_dim, name='encoder_output')(x)
        
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        
        ### THE DECODER
        decoder_input = tf.keras.layers.Input(shape=(self.z_dim,), name='decoder_input')
        x = tf.keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
        x = tf.keras.layers.Reshape(shape_before_flattening)(x)
        
        for i in range(self.n_layers_decoder):
            x =   tf.keras.layers.Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = 'same',
                name = 'decoder_conv_t_' + str(i)
            )(x)
          
            if i < self.n_layers_decoder - 1:
                x = tf.keras.layers.LeakyReLU()(x)
                if self.use_batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x)
                if self.use_dropout:
                    x = tf.keras.layers.Dropout(rate=0.25)(x)
            else:
                x = tf.keras.layers.Activation('sigmoid')(x)
              
        decoder_output = x
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)
        
        ### THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        
        self.model = tf.keras.models.Model(model_input, model_output)


    def save(self, folder):
        self.save_params(os.path.join(folder, 'params.pkl'))
        self.save_weights(os.path.join(folder, 'weights/weights.h5'))


    @staticmethod
    def load(folder, epoch=None):   # AutoEncoder.load(folder)
        params = AutoEncoder.load_params(os.path.join(folder, 'params.pkl'))
        AE = AutoEncoder(*params)
        if epoch is None:
            AE.model.load_weights(os.path.join(folder, 'weights/weights.h5'))
        else:
            AE.model.load_weights(os.path.join(folder, f'weights/weights_{epoch-1}.h5'))
            AE.epoch = epoch

        return AE


    def save_params(self, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        with open(filepath, 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
                self.epoch
            ], f)


    @staticmethod
    def load_params(filepath):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        return params


    def save_weights(self, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        self.model.save_weights(filepath)
        
        
    def load_weights(self, filepath):
        self.model.load_weights(filepath)


    def save_images(self, imgs, filepath):
        z_points = self.encoder.predict(imgs)
        reconst_imgs = self.decoder.predict(z_points)
        txts = [ f'{p[0]:.3f}, {p[1]:.3f}' for p in z_points ]
        AutoEncoder.showImages(imgs, reconst_imgs, txts, 1.4, 1.4, 0.5, filepath)
      

    @staticmethod
    def r_loss(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=[1,2,3])


    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss = AutoEncoder.r_loss)

        
    def train_with_fit(self,
               x_train,
               y_train,
               batch_size,
               epochs,
               run_folder='run/',
               validation_data=None
    ):
        history= self.model.fit(
            x_train,
            y_train,
            batch_size = batch_size,
            shuffle = True,
            initial_epoch = self.epoch,
            epochs = epochs,
            validation_data = validation_data
        )
        if self.epoch < epochs:
            self.epoch = epochs

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(os.path.join(run_folder,f'weights/weights_{self.epoch-1}.h5'))
            #idxs = np.random.choice(len(x_train), 10)
            #self.save_images(x_train[idxs], os.path.join(run_folder, f'images/image_{self.epoch-1}.png'))

        return history
        
        
    def train(self,
               x_train,
               y_train,
               batch_size = 32,
               epochs = 10,
               shuffle=False,
               run_folder='run/',
               optimizer=None,
               save_epoch_interval=100,
               validation_data = None
    ):
        start_time = datetime.datetime.now()
        steps = x_train.shape[0] // batch_size

        losses = []
        val_losses = []

        for epoch in range(self.epoch, epochs):
            epoch_loss = 0
            indices = tf.range(x_train.shape[0], dtype=tf.int32)
            if shuffle:
                indices = tf.random.shuffle(indices)
            x_ = x_train[indices]
            y_ = y_train[indices]
            
            for step in range(steps):
                start = batch_size * step
                end = start + batch_size

                with tf.GradientTape() as tape:
                    outputs = self.model(x_[start:end])
                    tmp_loss = AutoEncoder.r_loss(y_[start:end], outputs)

                grads = tape.gradient(tmp_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            epoch_loss = np.mean(tmp_loss)
            losses.append(epoch_loss)

            val_str = ''
            if validation_data != None:
                x_val, y_val = validation_data
                outputs_val = self.model(x_val)
                val_loss = np.mean(AutoEncoder.r_loss(y_val, outputs_val))
                val_str = f'val loss: {val_loss:.4f}  '
                val_losses.append(val_loss)


            if (epoch+1) % save_epoch_interval == 0 and run_folder != None:
                self.save(run_folder)
                self.save_weights(os.path.join(run_folder,f'weights/weights_{self.epoch}.h5'))
                #idxs = np.random.choice(len(x_train), 10)
                #self.save_images(x_train[idxs], os.path.join(run_folder, f'images/image_{self.epoch}.png'))

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1}/{epochs} {steps} loss: {epoch_loss:.4f}  {val_str}{elapsed_time}')

            self.epoch += 1

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(os.path.join(run_folder,f'weights/weights_{self.epoch-1}.h5'))
            #idxs = np.random.choice(len(x_train), 10)
            #self.save_images(x_train[idxs], os.path.join(run_folder, f'images/image_{self.epoch-1}.png'))

        return losses, val_losses

    @staticmethod
    @tf.function
    def compute_loss_and_grads(model,x,y):
        with tf.GradientTape() as tape:
            outputs = model(x)
            tmp_loss = AutoEncoder.r_loss(y,outputs)
        grads = tape.gradient(tmp_loss, model.trainable_variables)
        return tmp_loss, grads


    def train_tf(self,
               x_train,
               y_train,
               batch_size = 32,
               epochs = 10,
               shuffle=False,
               run_folder='run/',
               optimizer=None,
               save_epoch_interval=100,
               validation_data = None
    ):
        start_time = datetime.datetime.now()
        steps = x_train.shape[0] // batch_size

        losses = []
        val_losses = []

        for epoch in range(self.epoch, epochs):
            epoch_loss = 0
            indices = tf.range(x_train.shape[0], dtype=tf.int32)
            if shuffle:
                indices = tf.random.shuffle(indices)
            x_ = x_train[indices]
            y_ = y_train[indices]

            step_losses = []
            for step in range(steps):
                start = batch_size * step
                end = start + batch_size

                tmp_loss, grads = AutoEncoder.compute_loss_and_grads(self.model, x_[start:end], y_[start:end])
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                step_losses.append(np.mean(tmp_loss))

            epoch_loss = np.mean(step_losses)
            losses.append(epoch_loss)

            val_str = ''
            if validation_data != None:
                x_val, y_val = validation_data
                outputs_val = self.model(x_val)
                val_loss = np.mean(AutoEncoder.r_loss(y_val, outputs_val))
                val_str = f'val loss: {val_loss:.4f}  '
                val_losses.append(val_loss)


            if (epoch+1) % save_epoch_interval == 0 and run_folder != None:
                self.save(run_folder)
                self.save_weights(os.path.join(run_folder,f'weights/weights_{self.epoch}.h5'))
                #idxs = np.random.choice(len(x_train), 10)
                #self.save_images(x_train[idxs], os.path.join(run_folder, f'images/image_{self.epoch}.png'))

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1}/{epochs} {steps} loss: {epoch_loss:.4f}  {val_str}{elapsed_time}')

            self.epoch += 1

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(os.path.join(run_folder,f'weights/weights_{self.epoch-1}.h5'))
            #idxs = np.random.choice(len(x_train), 10)
            #self.save_images(x_train[idxs], os.path.join(run_folder, f'images/image_{self.epoch-1}.png'))

        return losses, val_losses


    @staticmethod
    def showImages(imgs1, imgs2, txts, w, h, vskip=0.5, filepath=None):
        n = len(imgs1)
        fig, ax = plt.subplots(2, n, figsize=(w * n, (2+vskip) * h))
        for i in range(n):
            if n == 1:
                axis = ax[0]
            else:
                axis = ax[0][i]
            img = imgs1[i].squeeze()
            axis.imshow(img, cmap='gray_r')
            axis.axis('off')

            axis.text(0.5, -0.35, txts[i], fontsize=10, ha='center', transform=axis.transAxes)

            if n == 1:
                axis = ax[1]
            else:
                axis = ax[1][i]
            img2 = imgs2[i].squeeze()
            axis.imshow(img2, cmap='gray_r')
            axis.axis('off')

        if not filepath is None:
            dpath, fname = os.path.split(filepath)
            if dpath != '' and not os.path.exists(dpath):
                os.makedirs(dpath)
            fig.savefig(filepath, dpi=600)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_history(vals, labels):
        colors = ['red', 'blue', 'green', 'orange', 'black']
        n = len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(9,4))
        for i in range(n):
            ax.plot(vals[i], c=colors[i], label=labels[i])
        ax.legend(loc='upper right')
        ax.set_xlabel('epochs')
        # ax[0].set_ylabel('loss')
        
        plt.show()
