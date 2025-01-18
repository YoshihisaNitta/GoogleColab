import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle as pkl
import datetime


class WGAN():
    def __init__(
        self,
        input_dim,
        critic_conv_filters,
        critic_conv_kernel_size,
        critic_conv_strides,
        critic_batch_norm_momentum,
        critic_activation,
        critic_dropout_rate,
        critic_learning_rate,
        generator_initial_dense_layer_size,
        generator_upsample,
        generator_conv_filters,
        generator_conv_kernel_size,
        generator_conv_strides,
        generator_batch_norm_momentum,
        generator_activation,
        generator_dropout_rate,
        generator_learning_rate,
        optimizer,
        z_dim,
        epoch = 0,
        d_losses = [],
        g_losses = []
    ):
        self.name = 'wgan'
        self.input_dim = input_dim
        
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_conv_strides = critic_conv_strides
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate
        
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        
        self.optimizer = optimizer
        self.z_dim = z_dim

        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len (generator_conv_filters)

        self.weight_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

        self.epoch = epoch

        self.d_losses = d_losses
        self.g_losses = g_losses


        self._build_critic()
        self._build_generator()

        self._build_adversarial()

    def wasserstein(self, y_true, y_pred):
        return - tf.keras.backend.mean(y_true * y_pred)
    
    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = tf.keras.layers.LeakyReLU(alpha = 0.2)
        else:
            layer = tf.keras.layers.Activation(activation)
        return layer

    def _build_critic(self):

        ### The Critic
        critic_input = tf.keras.layers.Input(shape=self.input_dim, name='critic_input')

        x = critic_input

        for i in range(self.n_layers_critic):
            x = tf.keras.layers.Conv2D(
                filters = self.critic_conv_filters[i],
                kernel_size = self.critic_conv_kernel_size[i],
                strides = self.critic_conv_strides[i],
                padding = 'same',
                name = 'critic_conv_' + str(i),
                kernel_initializer = self.weight_init
            )(x)
            if self.critic_batch_norm_momentum and i > 0:
                x = tf.keras.layers.BatchNormalization(momentum = self.critic_batch_norm_momentum)(x)
            x = self.get_activation(self.critic_activation)(x)
            if self.critic_dropout_rate:
                x = tf.keras.layers.Dropout(rate=self.critic_dropout_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        critic_output = tf.keras.layers.Dense(1, activation=None, kernel_initializer = self.weight_init)(x)
        self.critic = tf.keras.models.Model(critic_input, critic_output)

    def _build_generator(self):
        ### The Generator
        generator_input = tf.keras.layers.Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input
        x = tf.keras.layers.Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = tf.keras.layers.BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
        x = self.get_activation(self.generator_activation)(x)
        x = tf.keras.layers.Reshape(self.generator_initial_dense_layer_size)(x)
        if self.generator_dropout_rate:
            x = tf.keras.layers.Dropout(rate=self.generator_dropout_rate)(x)
        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                x = tf.keras.layers.UpSampling2D()(x)
                x = tf.keras.layers.Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    #strides=self.generator_conv_strides[i],  #[自分へのメモ]元のソースにはこの行はないので追加した。BUG ???
                    name='generator_conv_'+str(i),
                    kernel_initializer=self.weight_init
                )(x)
            else:
                x = tf.keras.layers.Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name='generator_conv_'+str(i),
                    kernel_initializer=self.weight_init
                )(x)

            if i < self.n_layers_generator -1:
                if self.generator_batch_norm_momentum:
                    x = tf.keras.layers.BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                x = tf.keras.layers.Activation('tanh')(x)

        generator_output = x
        self.generator = tf.keras.models.Model(generator_input, generator_output)

    def get_opti(self, lr):
        if self.optimizer == 'adam':
            opti = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimizer == 'rmsprop':
            opti = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opti = tf.keras.optimizers.Adam(learning_rate=lr)
        return opti

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):
        
        ### Compile critic
        self.critic.compile(
            optimizer=self.get_opti(self.critic_learning_rate),
            loss=self.wasserstein
        )
        
        ### Compile The Full GAN
        self.set_trainable(self.critic, False)
        model_input = tf.keras.layers.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.critic(self.generator(model_input))
        self.model = tf.keras.models.Model(model_input, model_output)

        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, clip_threshold, using_generator):
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]    # next() function generates images ???
            if true.imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real = self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)

        # for l in self.critic.layers:
        #     weights = l.get_weights()
        #     if 'batch_normalization' in l.get_config()['name']:
        #         # weights = [np.clip(w, -0.01, 0.01) for w in weights[:2]] + weights[2:]
        #         pass
        #    else:
        #         weights = [np.clip(w, -0.01, 0.01) for w in weights]
        #     l.set_weights(weights)

        return [d_loss, d_loss_real, d_loss_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=1000, n_critic=5, clip_threshold=0.01, using_generator=False):
        start_time = datetime.datetime.now()

        for epoch in range(self.epoch, epochs):
            for _ in range(n_critic):
                d_loss = self.train_critic(x_train, batch_size, clip_threshold, using_generator)
            g_loss = self.train_generator(batch_size)

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1} [D loss: {d_loss[0]:.3f} R {d_loss[1]:.3f} F {d_loss[2]:.3f}] [G loss: {g_loss:.3f}]  {elapsed_time}')

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            self.epoch += 1

            if self.epoch % print_every_n_batches == 0:
                self.save(run_folder, self.epoch)
                self.save(run_folder)

        self.save(run_folder, self.epoch)
        self.save(run_folder)


    def save(self, folder, epoch=None):
        self.save_params(folder, epoch)
        self.save_weights(folder,epoch)


    @staticmethod
    def load(folder, epoch=None):
        params = WGAN.load_params(folder, epoch)
        gan = WGAN(*params)
        gan.load_weights(folder, epoch)

        return gan


    def save_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.save_model_weights(self.critic, os.path.join(run_folder, 'weights/critic-weights.h5'))
            self.save_model_weights(self.generator, os.path.join(run_folder, 'weights/generator-weights.h5'))
            self.save_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))
        else:
            self.save_model_weights(self.critic, os.path.join(run_folder, f'weights/critic-weights_{epoch}.h5'))
            self.save_model_weights(self.generator, os.path.join(run_folder, f'weights/generator-weights_{epoch}.h5'))
            self.save_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))


    def load_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.load_model_weights(self.critic, os.path.join(run_folder, 'weights/critic-weights.h5'))
            self.load_model_weights(self.generator, os.path.join(run_folder, 'weights/generator-weights.h5'))
            self.load_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))
        else:
            self.load_model_weights(self.critic, os.path.join(run_folder, f'weights/critic-weights_{epoch}.h5'))
            self.load_model_weights(self.generator, os.path.join(run_folder, f'weights/generator-weights_{epoch}.h5'))
            self.load_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))


    def save_model_weights(self, model, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        model.save_weights(filepath)


    def load_model_weights(self, model, filepath):
        model.load_weights(filepath)


    def save_params(self, folder, epoch=None):
        if epoch is None:
            filepath = os.path.join(folder, 'params.pkl')
        else:
            filepath = os.path.join(folder, f'params_{epoch}.pkl')

        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)

        with open(os.path.join(filepath), 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.critic_conv_filters,
                self.critic_conv_kernel_size,
                self.critic_conv_strides,
                self.critic_batch_norm_momentum,
                self.critic_activation,
                self.critic_dropout_rate,
                self.critic_learning_rate,
                self.generator_initial_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate,
                self.optimizer,
                self.z_dim,
                self.epoch,
                self.d_losses,
                self.g_losses
            ], f)


    @staticmethod
    def load_params(folder, epoch=None):
        if epoch is None:
            filepath = os.path.join(folder, 'params.pkl')
        else:
            filepath = os.path.join(folder, f'params_{epoch}.pkl')

        with open(filepath, 'rb') as f:
            params = pkl.load(f)
        return params


    def generate_images(self, noise=[], n=10):
        if len(noise) == 0:
            noise = np.random.normal(0, 1, (n, self.z_dim))

        imgs = self.generator.predict(noise)   # [-1.0,1.0]
        imgs = 0.5 * (imgs + 1)        # [0.0, 1.0]
        imgs = np.clip(imgs, 0, 1)

        return imgs


    @staticmethod
    def showImages(xs, rows=-1, cols=-1, w=2.8, h=2.8, filepath=None):
        N = len(xs)
        if rows < 0: rows = 1
        if cols < 0: cols = (N + rows - 1) // rows
        fig, ax = plt.subplots(rows, cols, figsize=(w*cols, h*rows))
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if rows == 1 and cols == 1:
                    axis = ax
                elif cols == 1:
                    axis = ax[row]
                elif rows == 1:
                    axis = ax[col]
                else:
                    axis = ax[row][col]

                if idx < N:
                    axis.imshow(xs[idx], cmap='gray')
                axis.axis('off')
                idx += 1

        if not filepath is None:
            dpath, fname = os.path.split(filepath)
            if dpath != '' and not os.path.exists(dpath):
                os.makedirs(dpath)
            fig.savefig(filepath, dpi=600)
            plt.close()
        else:
            plt.show()


    def showLoss(self, xlim=[], ylim=[]):
        d = np.array(self.d_losses)
        g = np.array(self.g_losses)
        d_loss = d[:, 0]
        d_loss_real = d[:, 1]
        d_loss_fake = d[:, 2]
        g_loss = g
        WGAN.plot_history([d_loss, d_loss_real, d_loss_fake, g_loss], ['d_loss', 'd_loss_real', 'd_loss_fake', 'g_loss'],xlim,ylim)


    @staticmethod
    def plot_history(vals, labels, xlim=[], ylim=[]):
        colors = ['red', 'blue', 'green', 'orange', 'black', 'pink']
        n = len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(9,4))
        for i in range(n):
            ax.plot(vals[i], c=colors[i], label=labels[i])
        ax.legend(loc='upper right')
        ax.set_xlabel('epochs')
        # ax.set_ylabel('loss')

        if xlim != []:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim != []:
            ax.set_ylim(ylim[0], ylim[1])
        
        plt.show()
