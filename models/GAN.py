import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle as pkl
import datetime


class GAN():
    def __init__(
        self,
        input_dim,
        discriminator_conv_filters,
        discriminator_conv_kernel_size,
        discriminator_conv_strides,
        discriminator_batch_norm_momentum,
        discriminator_activation,
        discriminator_dropout_rate,
        discriminator_learning_rate,
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
        self.name = 'gan'
        self.input_dim = input_dim
        
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        
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

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len (generator_conv_filters)

        self.weight_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

        self.epoch = epoch

        self.d_losses = d_losses
        self.g_losses = g_losses

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()


    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = tf.keras.layers.LeakyReLU(alpha = 0.2)
        else:
            layer = tf.keras.layers.Activation(activation)
        return layer


    def _build_discriminator(self):

        ### The Discriminator
        discriminator_input = tf.keras.layers.Input(shape=self.input_dim, name='discriminator_input')

        x = discriminator_input

        for i in range(self.n_layers_discriminator):
            x = tf.keras.layers.Conv2D(
                filters = self.discriminator_conv_filters[i],
                kernel_size = self.discriminator_conv_kernel_size[i],
                strides = self.discriminator_conv_strides[i],
                padding = 'same',
                name = 'discriminator_conv_' + str(i),
                kernel_initializer = self.weight_init
            )(x)
            if self.discriminator_batch_norm_momentum and i > 0:
                x = tf.keras.layers.BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)
            x = self.get_activation(self.discriminator_activation)(x)
            if self.discriminator_dropout_rate:
                x = tf.keras.layers.Dropout(rate=self.discriminator_dropout_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)
        self.discriminator = tf.keras.models.Model(discriminator_input, discriminator_output)


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
                    strides=self.generator_conv_strides[i],
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
        
        ### Compile Discriminator
        self.discriminator.compile(
            optimizer=self.get_opti(self.discriminator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        ### Compile The Full GAN
        self.set_trainable(self.discriminator, False)
        model_input = tf.keras.layers.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = tf.keras.models.Model(model_input, model_output)

        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'],
            experimental_run_tf_function=False
        )

        self.set_trainable(self.discriminator, True)


    def train_discriminator(self, x_train, batch_size, using_generator):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:        # when x_train is DataGenerator
            true_imgs, _ = next(x_train)
            if true.imgs.shape[0] != batch_size:
                true_imgs, _ = next(x_train)
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]


    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=1000, using_generator=False):
        start_time = datetime.datetime.now()

        for epoch in range(self.epoch, epochs):
            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{epoch+1} [D loss: {d[0]:.3f} R {d[1]:.3f} F {d[2]:.3f}][D acc: {d[3]:.3f} R {d[4]:.3f} F {d[5]:.3f}][G loss: {g[0]:.3f} acc: {g[1]:.3f}] {elapsed_time}')

            self.d_losses.append(d)
            self.g_losses.append(g)

            self.epoch += 1

            if self.epoch % print_every_n_batches == 0:
                self.save(run_folder, self.epoch);
                self.save(run_folder);

        self.save(run_folder, self.epoch)
        self.save(run_folder)


    def save(self, folder, epoch=None):
        if epoch is None:
            self.save_params(os.path.join(folder, 'params.pkl'))
            #self.save_model(folder)
            self.save_weights(folder)
        else:
            self.save_params(os.path.join(folder, f'params_{epoch}.pkl'))
            self.save_weights(folder,epoch)


    @staticmethod
    def load(folder, epoch=None):
        if epoch is None:
            params = GAN.load_params(os.path.join(folder, 'params.pkl'))
            gan = GAN(*params)
            gan.load_weights(folder);
        else:
            params = GAN.load_params(os.path.join(folder, f'params_{epoch}.pkl'))
            gan = GAN(*params)
            gan.load_weights(folder, epoch)
        return gan


    def save_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.save_model_weights(self.discriminator, os.path.join(run_folder, 'weights/discriminator-weights.h5'))
            self.save_model_weights(self.generator, os.path.join(run_folder, 'weights/generator-weights.h5'))
            self.save_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))
        else:
            self.save_model_weights(self.discriminator, os.path.join(run_folder, f'weights/discriminator-weights_{epoch}.h5'))
            self.save_model_weights(self.generator, os.path.join(run_folder, f'weights/generator-weights_{epoch}.h5'))
            self.save_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))


    def load_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.load_model_weights(self.discriminator, os.path.join(run_folder, 'weights/discriminator-weights.h5'))
            self.load_model_weights(self.generator, os.path.join(run_folder, 'weights/generator-weights.h5'))
            self.load_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))
        else:
            self.load_model_weights(self.discriminator, os.path.join(run_folder, f'weights/discriminator-weights_{epoch}.h5'))
            self.load_model_weights(self.generator, os.path.join(run_folder, f'weights/generator-weights_{epoch}.h5'))
            self.load_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))


    def save_model_weights(self, model, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        model.save_weights(filepath)


    def load_model_weights(self, model, filepath):
        model.load_weights(filepath)


    def save_params(self, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        with open(filepath, 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.discriminator_conv_filters,
                self.discriminator_conv_kernel_size,
                self.discriminator_conv_strides,
                self.discriminator_batch_norm_momentum,
                self.discriminator_activation,
                self.discriminator_dropout_rate,
                self.discriminator_learning_rate,
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
    def load_params(filepath):
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


    def showLoss(self):
        d = np.array(self.d_losses)
        g = np.array(self.g_losses)
        d_loss = d[:, 0]
        d_loss_real = d[:, 1]
        d_loss_fake = d[:, 2]
        g_loss = g[:,0]
        GAN.plot_history([d_loss, d_loss_real, d_loss_fake, g_loss], ['d_loss', 'd_loss_real', 'd_loss_fake', 'g_loss'])


    def showAcc(self):
        d = np.array(self.d_losses)
        g = np.array(self.g_losses)
        d_acc = d[:, 3]
        d_acc_real = d[:, 4]
        d_acc_fake = d[:, 5]
        g_acc = g[:,1]
        GAN.plot_history([d_acc, d_acc_real, d_acc_fake, g_acc], ['d_acc', 'd_acc_real', 'd_acc_fake', 'g_acc'])


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
