import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
from functools import partial

import matplotlib.pyplot as plt

import os
import pickle as pkl
import datetime


def grad(y,x):
    V = tf.keras.layers.Lambda(
        lambda z: tf.keras.backend.gradients(z[0],z[1]),
        output_shape=[1]
        )([y,x])
    return V


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    '''Provides a (random) weighted average between real and generated image samples'''
    def call(self, inputs):
        alpha = tf.keras.backend.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
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
        grad_weight,  # wgangp
        z_dim,
        batch_size,    # wgangp
        epoch = 0,
        d_losses = [],
        g_losses = []
    ):
        self.name = 'wgangp'
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
        self.grad_weight = grad_weight
        self.batch_size = batch_size

        self.epoch = epoch

        self.d_losses = d_losses
        self.g_losses = g_losses


        self._build_critic()
        self._build_generator()

        self._build_adversarial()


    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        '''
        Computes gradient penalty based on prediction and weighted real/fake samples
        '''
        gradients = grad(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = tf.keras.backend.square(gradients)
        # ... summing over the rows ...
        gradients_sqr_sum = tf.keras.backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # ... and sqrt
        gradient_12_norm = tf.keras.backend.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - || grad|| )^2 still for each single sample
        gradient_penalty = tf.keras.backend.square(1 - gradient_12_norm)
        # return the mean as loss over all the batch samples
        return tf.keras.backend.mean(gradient_penalty)


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
                    #strides=self.generator_conv_strides[i],  # [自分へのメモ] 元ソースではなぜか stride は使っていない。BUG? ここでは元ソースの通りにstridesは指定しないことにする。
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


    def _build_adversarial(self):   # WGANと比較して、この作りが大幅に変わるようだ

        # --------------------------------------------
        # Construct Computational Graph for the Critic
        # --------------------------------------------

        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = tf.keras.layers.Input(shape=self.input_dim)

        # Fake Image
        z_disc = tf.keras.layers.Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # critic determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional 'interpolated_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = tf.keras.models.Model(
            inputs=[real_img, z_disc], 
            outputs=[valid, fake, validity_interpolated]
        )
        
        ### Compile critic
        # When the Model has multiple outputs, you can use different losses for each output by passing a dictionary or list to loss.
        # Minimize the sum of individual losses weighted by the loss_weights factor.
        # Modelが複数のoutputを持つ場合は、lossに辞書かリストを渡すことで、各outputに異なる損失を用いることができる。
        # Model によって最小化されるのは loss_weights 係数で重みづけされた個々の損失の加重合計である。
        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
            optimizer=self.get_opti(self.critic_learning_rate),
            loss_weights = [1, 1, self.grad_weight]
        )

        #--------------------------------------------
        # Construct Computational Graph for Generator
        #--------------------------------------------

        # For the generator, the critic's layers are freezed
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        model_input = tf.keras.layers.Input(shape=(self.z_dim,))
        # Generate images based of noise
        img = self.generator(model_input)
        # critic (Discriminator) determines validity
        model_output = self.critic(img)
        # Defines generator model
        self.model = tf.keras.models.Model(model_input, model_output)

        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)


    def train_critic(self, x_train, batch_size, using_generator):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32) # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        d_loss = self.critic_model.train_on_batch([true_imgs, noise], [valid, fake, dummy])

        return d_loss


    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, n_critic=5, using_generator=False):
        start_time = datetime.datetime.now()

        for epoch in range(self.epoch, epochs):

            if (epoch+1) % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic
                
            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)
            g_loss = self.train_generator(batch_size)

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1} ({critic_loops}, 1) [D loss: {d_loss[0]:.3f} R {d_loss[1]:.3f} F {d_loss[2]:.3f} G {d_loss[3]:.3f}][G loss: {g_loss:.3f}]  {elapsed_time}')

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
        params = WGANGP.load_params(folder, epoch)
        gan = WGANGP(*params)
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
                self.grad_weight,
                self.z_dim,
                self.batch_size,
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
        d_loss_gen = d[:, 3]
        g_loss = g
        WGANGP.plot_history(
            [d_loss, d_loss_real, d_loss_fake, d_loss_gen, g_loss],
            ['d_loss', 'd_loss_real', 'd_loss_fake', 'd_loss_gen', 'g_loss'],
            xlim,
            ylim)


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
