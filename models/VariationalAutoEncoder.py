import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
import datetime

class Sampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0., stddev=1.)
        return mu + tf.keras.backend.exp(log_var / 2) * epsilon


class VAEModel(tf.keras.models.Model):
    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor


    @tf.function
    def loss_fn(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.square(x - reconstruction), axis=[1,2,3]
        ) * self.r_loss_factor
        kl_loss = tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis = 1
        ) * (-0.5)
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss


    @tf.function
    def compute_loss_and_grads(self, x):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.loss_fn(x)
        grads = tape.gradient(total_loss, self.trainable_weights)
        return total_loss, reconstruction_loss, kl_loss, grads


    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        total_loss, reconstruction_loss, kl_loss, grads = self.compute_loss_and_grads(data)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": tf.math.reduce_mean(total_loss),
            "reconstruction_loss": tf.math.reduce_mean(reconstruction_loss),
            "kl_loss": tf.math.reduce_mean(kl_loss),
        }

    def call(self,inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)


class VariationalAutoEncoder():
    def __init__(self, 
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_strides,
                 z_dim,
                 r_loss_factor,   ### added
                 use_batch_norm = False,
                 use_dropout = False,
                 epoch = 0
                ):
        self.name = 'variational_autoencoder'
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.r_loss_factor = r_loss_factor   ### added
            
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
            x = conv_layer = tf.keras.layers.Conv2D(
                filters = self.encoder_conv_filters[i],
                kernel_size = self.encoder_conv_kernel_size[i],
                strides = self.encoder_conv_strides[i],
                padding  = 'same',
                name = 'encoder_conv_' + str(i)
            )(x)

            if self.use_batch_norm:                                ### The order of layers is opposite to AutoEncoder
                x = tf.keras.layers.BatchNormalization()(x)        ###   AE: LeakyReLU -> BatchNorm
            x = tf.keras.layers.LeakyReLU()(x)                     ###   VAE: BatchNorm -> LeakyReLU
            
            if self.use_dropout:
                x = tf.keras.layers.Dropout(rate = 0.25)(x)
        
        shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
        
        x = tf.keras.layers.Flatten()(x)
        
        self.mu = tf.keras.layers.Dense(self.z_dim, name='mu')(x)
        self.log_var = tf.keras.layers.Dense(self.z_dim, name='log_var')(x) 
        self.z = Sampling(name='encoder_output')([self.mu, self.log_var])
        
        self.encoder = tf.keras.models.Model(encoder_input, [self.mu, self.log_var, self.z], name='encoder')
        
        
        ### THE DECODER
        decoder_input = tf.keras.layers.Input(shape=(self.z_dim,), name='decoder_input')
        x = decoder_input
        x = tf.keras.layers.Dense(np.prod(shape_before_flattening))(x)
        x = tf.keras.layers.Reshape(shape_before_flattening)(x)
        
        for i in range(self.n_layers_decoder):
            x = conv_t_layer =   tf.keras.layers.Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = 'same',
                name = 'decoder_conv_t_' + str(i)
            )(x)
            
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:                           ### The order of layers is opposite to AutoEncoder
                    x = tf.keras.layers.BatchNormalization()(x)   ###     AE: LeakyReLU -> BatchNorm
                x = tf.keras.layers.LeakyReLU()(x)                ###      VAE: BatchNorm -> LeakyReLU                
                if self.use_dropout:
                    x = tf.keras.layers.Dropout(rate=0.25)(x)
            else:
                x = tf.keras.layers.Activation('sigmoid')(x)
       
        decoder_output = x
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output, name='decoder')  ### added (name)
        
        ### THE FULL AUTOENCODER
        self.model = VAEModel(self.encoder, self.decoder, self.r_loss_factor)
        
        
    def save(self, folder):
        self.save_params(os.path.join(folder, 'params.pkl'))
        self.save_weights(folder)


    @staticmethod
    def load(folder, epoch=None):  # VariationalAutoEncoder.load(folder)
        params = VariationalAutoEncoder.load_params(os.path.join(folder, 'params.pkl'))
        VAE = VariationalAutoEncoder(*params)
        if epoch is None:
            VAE.load_weights(folder)
        else:
            VAE.load_weights(folder, epoch-1)
            VAE.epoch = epoch
        return VAE

        
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
                self.r_loss_factor,
                self.use_batch_norm,
                self.use_dropout,
                self.epoch
            ], f)


    @staticmethod
    def load_params(filepath):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        return params


    def save_weights(self, folder, epoch=None):
        if epoch is None:
            self.save_model_weights(self.encoder, os.path.join(folder, f'weights/encoder-weights.h5'))
            self.save_model_weights(self.decoder, os.path.join(folder, f'weights/decoder-weights.h5'))
        else:
            self.save_model_weights(self.encoder, os.path.join(folder, f'weights/encoder-weights_{epoch}.h5'))
            self.save_model_weights(self.decoder, os.path.join(folder, f'weights/decoder-weights_{epoch}.h5'))


    def save_model_weights(self, model, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        model.save_weights(filepath)


    def load_weights(self, folder, epoch=None):
        if epoch is None:
            self.encoder.load_weights(os.path.join(folder, f'weights/encoder-weights.h5'))
            self.decoder.load_weights(os.path.join(folder, f'weights/decoder-weights.h5'))
        else:
            self.encoder.load_weights(os.path.join(folder, f'weights/encoder-weights_{epoch}.h5'))
            self.decoder.load_weights(os.path.join(folder, f'weights/decoder-weights_{epoch}.h5'))


    def save_images(self, imgs, filepath):
        z_mean, z_log_var, z = self.encoder.predict(imgs)
        reconst_imgs = self.decoder.predict(z)
        txts = [ f'{p[0]:.3f}, {p[1]:.3f}' for p in z ]
        AutoEncoder.showImages(imgs, reconst_imgs, txts, 1.4, 1.4, 0.5, filepath)
      

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer)     # CAUTION!!!: loss(y_true, y_pred) function is not specified.
        
        
    def train_with_fit(
            self,
            x_train,
            batch_size,
            epochs,
            run_folder='run/'
    ):
        history = self.model.fit(
            x_train,
            x_train,
            batch_size = batch_size,
            shuffle=True,
            initial_epoch = self.epoch,
            epochs = epochs
        )
        if (self.epoch < epochs):
            self.epoch = epochs

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(run_folder, self.epoch-1)
        
        return history


    def train_generator_with_fit(
            self,
            data_flow,
            epochs,
            run_folder='run/'
    ):
        history = self.model.fit(
            data_flow,
            initial_epoch = self.epoch,
            epochs = epochs
        )
        if (self.epoch < epochs):
            self.epoch = epochs

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(run_folder, self.epoch-1)
        
        return history


    def train_tf(
            self,
            x_train,
            batch_size = 32,
            epochs = 10,
            shuffle = False,
            run_folder = 'run/',
            optimizer = None,
            save_epoch_interval = 100,
            validation_data = None
    ):
        start_time = datetime.datetime.now()
        steps = x_train.shape[0] // batch_size

        total_losses = []
        reconstruction_losses = []
        kl_losses = []

        val_total_losses = []
        val_reconstruction_losses = []
        val_kl_losses = []

        for epoch in range(self.epoch, epochs):
            epoch_loss = 0
            indices = tf.range(x_train.shape[0], dtype=tf.int32)
            if shuffle:
                indices = tf.random.shuffle(indices)
            x_ = x_train[indices]

            step_total_losses = []
            step_reconstruction_losses = []
            step_kl_losses = []
            for step in range(steps):
                start = batch_size * step
                end = start + batch_size

                total_loss, reconstruction_loss, kl_loss, grads = self.model.compute_loss_and_grads(x_[start:end])
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                step_total_losses.append(np.mean(total_loss))
                step_reconstruction_losses.append(np.mean(reconstruction_loss))
                step_kl_losses.append(np.mean(kl_loss))
            
            epoch_total_loss = np.mean(step_total_losses)
            epoch_reconstruction_loss = np.mean(step_reconstruction_losses)
            epoch_kl_loss = np.mean(step_kl_losses)

            total_losses.append(epoch_total_loss)
            reconstruction_losses.append(epoch_reconstruction_loss)
            kl_losses.append(epoch_kl_loss)

            val_str = ''
            if not validation_data is None:
                x_val = validation_data
                tl, rl, kl = self.model.loss_fn(x_val)
                val_tl = np.mean(tl)
                val_rl = np.mean(rl)
                val_kl = np.mean(kl)
                val_total_losses.append(val_tl)
                val_reconstruction_losses.append(val_rl)
                val_kl_losses.append(val_kl)
                val_str = f'val loss total {val_tl:.3f} reconstruction {val_rl:.3f} kl {val_kl:.3f} '

            if (epoch+1) % save_epoch_interval == 0 and run_folder != None:
                self.save(run_folder)
                self.save_weights(run_folder, self.epoch)

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1}/{epochs} {steps} loss: total {epoch_total_loss:.3f} reconstruction {epoch_reconstruction_loss:.3f} kl {epoch_kl_loss:.3f} {val_str}{elapsed_time}')

            self.epoch += 1

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(run_folder, self.epoch-1)

        dic = { 'loss' : total_losses, 'reconstruction_loss' : reconstruction_losses, 'kl_loss' : kl_losses }
        if not validation_data is None:
            dic['val_loss'] = val_total_losses
            dic['val_reconstruction_loss'] = val_reconstruction_losses
            dic['val_kl_loss'] = val_kl_losses

        return dic
            

    def train_tf_generator(
            self,
            data_flow,
            epochs = 10,
            run_folder = 'run/',
            optimizer = None,
            save_epoch_interval = 100,
            validation_data_flow = None
    ):
        start_time = datetime.datetime.now()
        steps = len(data_flow)

        total_losses = []
        reconstruction_losses = []
        kl_losses = []

        val_total_losses = []
        val_reconstruction_losses = []
        val_kl_losses = []

        for epoch in range(self.epoch, epochs):
            epoch_loss = 0

            step_total_losses = []
            step_reconstruction_losses = []
            step_kl_losses = []

            for step in range(steps):
                x, _ = next(data_flow)

                total_loss, reconstruction_loss, kl_loss, grads = self.model.compute_loss_and_grads(x)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                step_total_losses.append(np.mean(total_loss))
                step_reconstruction_losses.append(np.mean(reconstruction_loss))
                step_kl_losses.append(np.mean(kl_loss))
            
            epoch_total_loss = np.mean(step_total_losses)
            epoch_reconstruction_loss = np.mean(step_reconstruction_losses)
            epoch_kl_loss = np.mean(step_kl_losses)

            total_losses.append(epoch_total_loss)
            reconstruction_losses.append(epoch_reconstruction_loss)
            kl_losses.append(epoch_kl_loss)

            val_str = ''
            if not validation_data_flow is None:
                step_val_tl = []
                step_val_rl = []
                step_val_kl = []
                for i in range(len(validation_data_flow)):
                    x, _ = next(validation_data_flow)
                    tl, rl, kl = self.model.loss_fn(x)
                    step_val_tl.append(np.mean(tl))
                    step_val_rl.append(np.mean(rl))
                    step_val_kl.append(np.mean(kl))
                val_tl = np.mean(step_val_tl)
                val_rl = np.mean(step_val_rl)
                val_kl = np.mean(step_val_kl)
                val_total_losses.append(val_tl)
                val_reconstruction_losses.append(val_rl)
                val_kl_losses.append(val_kl)
                val_str = f'val loss total {val_tl:.3f} reconstruction {val_rl:.3f} kl {val_kl:.3f} '

            if (epoch+1) % save_epoch_interval == 0 and run_folder != None:
                self.save(run_folder)
                self.save_weights(run_folder, self.epoch)

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1}/{epochs} {steps} loss: total {epoch_total_loss:.3f} reconstruction {epoch_reconstruction_loss:.3f} kl {epoch_kl_loss:.3f} {val_str}{elapsed_time}')

            self.epoch += 1

        if run_folder != None:
            self.save(run_folder)
            self.save_weights(run_folder, self.epoch-1)

        dic = { 'loss' : total_losses, 'reconstruction_loss' : reconstruction_losses, 'kl_loss' : kl_losses }
        if not validation_data_flow is None:
            dic['val_loss'] = val_total_losses
            dic['val_reconstruction_loss'] = val_reconstruction_losses
            dic['val_kl_loss'] = val_kl_losses

        return dic


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
        colors = ['red', 'blue', 'green', 'orange', 'black', 'pink']
        n = len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(9,4))
        for i in range(n):
            ax.plot(vals[i], c=colors[i], label=labels[i])
        ax.legend(loc='upper right')
        ax.set_xlabel('epochs')
        # ax[0].set_ylabel('loss')
        
        plt.show()
