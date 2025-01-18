import tensorflow as tf
import tensorflow_addons as tf_addons
import numpy as np

import matplotlib.pyplot as plt

from collections import deque

import os
import glob
import pickle as pkl
import random
import datetime


################################################################################
# Data Loader
################################################################################
class SequenceDataset():
    def __init__(self, dir_paths, skip=0, batch_size=1, target_size=None):
        self.dir_paths = dir_paths
        self.skip = skip
        self.batch_size = batch_size
        self.target_size = target_size

        file_paths = []
        for dir in dir_paths:
            p = glob.glob(os.path.join(dir, "*"))
            file_paths.append(sorted(p))
        self.file_paths = file_paths

        self.len = sum(map(len, file_paths))
        self.index = 0

    def _build_tbl(self):
        a = [ (len(x) - batch_size +1) for x in self.file_paths ]


    def __len__(self):
        return self.len

        
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            if start == None: start = 0
            if stop == None: stop = self.__len__()
            if step == None:
                if start < stop:
                    step = 1
                elif start > stop:
                    step = -1
                else:
                    step = 0
            return np.array([self.__getitemInt__(i) for i in range(start, stop, step) ])
        else:
            return self.__getitemInt__(index)


    def __getitemInt__(self, index):
        path = self.paths[index % self.lenA]
        if self.unaligned:
            path_B = self.paths_B[np.random.choice(self.lenB, 1)]
        else:
            path_B = self.paths_B[index % self.lenB]
        img_A = np.array(tf.keras.utils.load_img(path_A, target_size = self.target_size))
        img_B = np.array(tf.keras.utils.load_img(path_B, target_size = self.target_size))
        img_A = (img_A.astype('float32') - 127.5) / 127.5
        img_B = (img_B.astype('float32') - 127.5) / 127.5
        return np.array([img_A, img_B])


    def getImage(self, path):
        img = np.array(tf.keras.utils.load_img(path, target_size = self.target_size))
        img = (img.astype('float32') - 127.5) / 127.5
        return np.array(img)


    def __next__(self):
        self.index += 1
        return self.__getitem__(self.index-1)


        

class PairDatasetSequence():
    def __init__(self, paths_A, paths_B, skip=0, batch_size= 1, target_size = None, unaligned=False):
        self.paths_A = np.array(paths_A)
        self.paths_B = np.array(paths_B)
        self.skip = skip
        self.target_size = target_size
        self.batch_size = batch_size
        self.unaligned = unaligned

        self.lenA = len(paths_A)
        self.lenB = len(paths_B)
        self.index = 0

    def __len__(self):
        return max(self.lenA, self.lenB)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            if start == None: start = 0
            if stop == None: stop = self.__len__()
            if step == None:
                if start < stop:
                    step = 1
                elif start > stop:
                    step = -1
                else:
                    step = 0
            return np.array([self.__getitemInt__(i) for i in range(start, stop, step) ])
        else:
            return self.__getitemInt__(index)

    def __getitemInt__(self, index):
        path_A = self.paths_A[index % self.lenA]
        if self.unaligned:
            path_B = self.paths_B[np.random.choice(self.lenB, 1)]
        else:
            path_B = self.paths_B[index % self.lenB]
        img_A = np.array(tf.keras.utils.load_img(path_A, target_size = self.target_size))
        img_B = np.array(tf.keras.utils.load_img(path_B, target_size = self.target_size))
        img_A = (img_A.astype('float32') - 127.5) / 127.5
        img_B = (img_B.astype('float32') - 127.5) / 127.5
        return np.array([img_A, img_B])

    def __next__(self):
        self.index += 1
        return self.__getitem__(self.index-1)



################################################################################
# Layer
################################################################################
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, s):
        '''
        If you are using "channels_last" configuration
        '''
        return (s[0], s[1]+2*self.padding[0], s[2]+2*self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


################################################################################
# Model
################################################################################
class CycleGAN():
    def __init__(
        self,
        input_dim,
        learning_rate,
        lambda_validation,
        lambda_reconstr,
        lambda_id,
        generator_type,
        gen_n_filters,
        disc_n_filters,
        buffer_max_length = 50,
        epoch = 0, 
        d_losses = [],
        g_losses = []
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.buffer_max_length = buffer_max_length
        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.epoch = epoch
        self.d_losses = d_losses
        self.g_losses = g_losses

        self.buffer_A = deque(maxlen=self.buffer_max_length)
        self.buffer_B = deque(maxlen=self.buffer_max_length)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch, patch, 1)
        
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.compile_models()


    def compile_models(self):
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.d_A.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(self.learning_rate, 0.5),
            metrics=['accuracy']
        )
        self.d_B.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(self.learning_rate, 0.5),
            metrics=['accuracy']
        )

        # Build the generators
        if self.generator_type == 'unet':
            self.g_AB = self.build_generator_unet()
            self.g_BA = self.build_generator_unet()
        else:
            self.g_AB = self.build_generator_resnet()
            self.g_BA = self.build_generator_resnet()

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Input images from both domains
        img_A = tf.keras.layers.Input(shape=self.img_shape)
        img_B = tf.keras.layers.Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)   # [self memo] ??? translate *A* from domainB to domainA 
        img_B_id = self.g_AB(img_B)   # [self memo] ??? translate *B* from domainA to domainB 
        
        # Discriminators determines validity of traslated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = tf.keras.models.Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id]
        )
        self.combined.compile(
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],  # Mean Squared Error, Mean Absolute Error
            loss_weights=[ self.lambda_validation, self.lambda_validation,
                         self.lambda_reconstr, self.lambda_reconstr,
                         self.lambda_id, self.lambda_id ],
            optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
        )
        self.d_A.trainable = True
        self.d_B.trainable = True


    def build_generator_unet(self):
        def downsample(layer_input, filters, f_size=4):
            d = tf.keras.layers.Conv2D(
                filters,
                kernel_size=f_size,
                strides=2,
                padding='same',
                kernel_initializer = self.weight_init  # [self memo] added by nitta
            )(layer_input)
            d = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = tf.keras.layers.Activation('relu')(d)
            return d
        def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            u = tf.keras.layers.Conv2D(
                filters, 
                kernel_size=f_size, 
                strides=1, 
                padding='same',
                kernel_initializer = self.weight_init  # [self memo] added by nitta
            )(u)
            u = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = tf.keras.layers.Activation('relu')(u)
            if dropout_rate:
                u = tf.keras.layers.Dropout(dropout_rate)(u)
            u = tf.keras.layers.Concatenate()([u, skip_input])
            return u
        # Image input
        img = tf.keras.layers.Input(shape=self.img_shape)
        # Downsampling
        d1 = downsample(img, self.gen_n_filters)
        d2 = downsample(d1, self.gen_n_filters*2)
        d3 = downsample(d2, self.gen_n_filters*4)
        d4 = downsample(d3, self.gen_n_filters*8)

        # Upsampling
        u1 = upsample(d4, d3, self.gen_n_filters*4)
        u2 = upsample(u1, d2, self.gen_n_filters*2)
        u3 = upsample(u2, d1, self.gen_n_filters)

        u4 = tf.keras.layers.UpSampling2D(size=2)(u3)
        output_img = tf.keras.layers.Conv2D(
            self.channels, 
            kernel_size=4,
            strides=1, 
            padding='same',
            activation='tanh',
            kernel_initializer = self.weight_init  # [self memo] added by nitta
        )(u4)

        return tf.keras.models.Model(img, output_img)


    def build_generator_resnet(self):
        def conv7s1(layer_input, filters, final):
            y = ReflectionPadding2D(padding=(3,3))(layer_input)
            y = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(7,7),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            if final:
                y = tf.keras.layers.Activation('tanh')(y)
            else:
                y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
                y = tf.keras.layers.Activation('relu')(y)
            return y

        def downsample(layer_input, filters):
            y = tf.keras.layers.Conv2D(
                filters, 
                kernel_size=(3,3), 
                strides=2, 
                padding='same',
                kernel_initializer = self.weight_init
            )(layer_input)
            y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = tf.keras.layers.Activation('relu')(y)
            return y

        def residual(layer_input, filters):
            shortcut = layer_input
            y = ReflectionPadding2D(padding=(1,1))(layer_input)
            y = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(3,3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = tf.keras.layers.Activation('relu')(y)
            y = ReflectionPadding2D(padding=(1,1))(y)
            y = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(3,3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
            return tf.keras.layers.add([shortcut, y])
          
        def upsample(layer_input, filters):
            y = tf.keras.layers.Conv2DTranspose(
                filters, 
                kernel_size=(3,3), 
                strides=2,
                padding='same',
                kernel_initializer=self.weight_init
            )(layer_input)
            y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = tf.keras.layers.Activation('relu')(y)
            return y

        # Image input
        img = tf.keras.layers.Input(shape=self.img_shape)

        y = img
        y = conv7s1(y, self.gen_n_filters, False)
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters)
        y = conv7s1(y, 3, True)
        output = y
        
        return tf.keras.models.Model(img, output)


    def build_discriminator(self):
        def conv4(layer_input, filters, stride=2, norm=True):
            y = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(4,4),
                strides=stride,
                padding='same',
                kernel_initializer = self.weight_init
              )(layer_input)
            if norm:
                y = tf_addons.layers.InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = tf.keras.layers.LeakyReLU(0.2)(y)
            return y

        img = tf.keras.layers.Input(shape=self.img_shape)
        y = conv4(img, self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, self.disc_n_filters*2, stride=2)
        y = conv4(y, self.disc_n_filters*4, stride=2)
        y = conv4(y, self.disc_n_filters*8, stride=1)
        output = tf.keras.layers.Conv2D(
            1,
            kernel_size=4,
            strides=1,
            padding='same',
            kernel_initializer=self.weight_init
        )(y)
        return tf.keras.models.Model(img, output)


    def train_discriminators(self, imgs_A, imgs_B, valid, fake):
        # Translate images to opposite domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        
        self.buffer_B.append(fake_B)
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A))) # random sampling without replacement 
        fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))
        
        # Train the discriminators (original images=real / translated = fake)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total discriminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0], 
            dA_loss[0], dA_loss_real[0], dA_loss_fake[0],
            dB_loss[0], dB_loss_real[0], dB_loss_fake[0],
            d_loss_total[1], 
            dA_loss[1], dA_loss_real[1], dA_loss_fake[1],
            dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )


    def train_generators(self, imgs_A, imgs_B, valid):
        # Train the generators
        return self.combined.train_on_batch(
          [imgs_A, imgs_B], 
          [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
        )


    def train(self, data_loader, epochs, batch_size=1, run_folder='./run', print_step_interval=100, save_epoch_interval=100):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truthes
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        steps = len(data_loader) // batch_size
        for epoch in range(self.epoch, epochs):
            step_d_losses = []
            step_g_losses = []
            for step in range(steps):
                start = step * batch_size
                end = start + batch_size
                pairs = data_loader[start:end]    # ((a,b), (a, b), ....)
                imgs_A, imgs_B = [], []
                for img_A, img_B in pairs:
                    imgs_A.append(img_A)
                    imgs_B.append(img_B)

                imgs_A = np.array(imgs_A)
                imgs_B = np.array(imgs_B)

                step_d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                step_g_loss = self.train_generators(imgs_A, imgs_B, valid)

                step_d_losses.append(step_d_loss)
                step_g_losses.append(step_g_loss)

                elapsed_time = datetime.datetime.now() - start_time
                if (step+1) % print_step_interval == 0:
                    print(f'Epoch {epoch+1}/{epochs} {step+1}/{steps} [D loss: {step_d_loss[0]:.3f} acc: {step_d_loss[7]:.3f}][G loss: {step_g_loss[0]:.3f} adv: {np.sum(step_g_loss[1:3]):.3f} recon: {np.sum(step_g_loss[3:5]):.3f} id: {np.sum(step_g_loss[5:7]):.3f} time: {elapsed_time:}')

            d_loss = np.mean(step_d_losses, axis=0)
            g_loss = np.mean(step_g_losses, axis=0)

            elapsed_time = datetime.datetime.now() - start_time

            elapsed_time = datetime.datetime.now() - start_time
            print(f'Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]:.3f} acc: {d_loss[7]:.3f}][G loss: {g_loss[0]:.3f} adv: {np.sum(g_loss[1:3]):.3f} recon: {np.sum(g_loss[3:5]):.3f} id: {np.sum(g_loss[5:7]):.3f} time: {elapsed_time:}')
                    
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            self.epoch += 1
            if (self.epoch) % save_epoch_interval == 0:
                self.save(run_folder, self.epoch)
                self.save(run_folder)

        self.save(run_folder, self.epoch)
        self.save(run_folder)


    def save(self, folder, epoch=None):
        self.save_params(folder, epoch)
        self.save_weights(folder,epoch)


    @staticmethod
    def load(folder, epoch=None):
        params = CycleGAN.load_params(folder, epoch)
        gan = CycleGAN(*params)
        gan.load_weights(folder, epoch)
        return gan


    def save_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.save_model_weights(self.combined, os.path.join(run_folder, 'weights/combined-weights.h5'))
            self.save_model_weights(self.d_A, os.path.join(run_folder, 'weights/d_A-weights.h5'))
            self.save_model_weights(self.d_B, os.path.join(run_folder, 'weights/d_B-weights.h5'))
            self.save_model_weights(self.g_AB, os.path.join(run_folder, 'weights/g_AB-weights.h5'))
            self.save_model_weights(self.g_BA, os.path.join(run_folder, 'weights/g_BA-weights.h5'))
        else:
            self.save_model_weights(self.combined, os.path.join(run_folder, f'weights/combined-weights_{epoch}.h5'))
            self.save_model_weights(self.d_A, os.path.join(run_folder, f'weights/d_A-weights_{epoch}.h5'))
            self.save_model_weights(self.d_B, os.path.join(run_folder, f'weights/d_B-weights_{epoch}.h5'))
            self.save_model_weights(self.g_AB, os.path.join(run_folder, f'weights/g_AB-weights_{epoch}.h5'))
            self.save_model_weights(self.g_BA, os.path.join(run_folder, f'weights/g_BA-weights_{epoch}.h5'))


    def load_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.load_model_weights(self.combined, os.path.join(run_folder, 'weights/combined-weights.h5'))
            self.load_model_weights(self.d_A, os.path.join(run_folder, 'weights/d_A-weights.h5'))
            self.load_model_weights(self.d_B, os.path.join(run_folder, 'weights/d_B-weights.h5'))
            self.load_model_weights(self.g_AB, os.path.join(run_folder, 'weights/g_AB-weights.h5'))
            self.load_model_weights(self.g_BA, os.path.join(run_folder, 'weights/g_BA-weights.h5'))
        else:
            self.load_model_weights(self.combined, os.path.join(run_folder, f'weights/combined-weights_{epoch}.h5'))
            self.load_model_weights(self.d_A, os.path.join(run_folder, f'weights/d_A-weights_{epoch}.h5'))
            self.load_model_weights(self.d_B, os.path.join(run_folder, f'weights/d_B-weights_{epoch}.h5'))
            self.load_model_weights(self.g_AB, os.path.join(run_folder, f'weights/g_AB-weights_{epoch}.h5'))
            self.load_model_weights(self.g_BA, os.path.join(run_folder, f'weights/g_BA-weights_{epoch}.h5'))


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

        with open(filepath, 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.learning_rate,
                self.lambda_validation,
                self.lambda_reconstr,
                self.lambda_id,
                self.generator_type,
                self.gen_n_filters,
                self.disc_n_filters,
                self.buffer_max_length,
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


    def generate_image(self, img_A, img_B):
        gen_A = self.generate_image_from_A(img_A)
        gen_B = self.generate_image_from_B(img_B)
        return np.concatenate([gen_A, gen_B], axis=0)


    def generate_image_from_A(self, img_A):
        fake_B = self.g_AB.predict(img_A)      # Translate images to the other domain
        reconstr_A = self.g_BA.predict(fake_B)  # Translate back to original domain
        id_A = self.g_BA.predict(img_A)    # ID the images
        return np.concatenate([img_A, fake_B, reconstr_A, id_A])


    def generate_image_from_B(self, img_B):
        fake_A = self.g_BA.predict(img_B)
        reconstr_B = self.g_AB.predict(fake_A)
        id_B = self.g_AB.predict(img_B)
        return np.concatenate([img_B, fake_A, reconstr_B, id_B])


    @staticmethod
    def showImages(imgs, trans, recon, idimg, w=2.8, h=2.8, filepath=None):
        N = len(imgs)
        M = len(imgs[0])
        titles = ['Original', 'Translated', 'Reconstructed', 'ID']

        fig, ax = plt.subplots(N, M, figsize=(w*M, h*N))
        for i in range(N):
            for j in range(M):
                ax[i][j].imshow(imgs[i][j])
                ax[i][j].set_title(title[j])
                ax[i][j].axis('off')

        if not filepath is None:
            dpath, fname = os.path.split(filepath)
            if dpath != '' and not os.path.exists(dpath):
                os.makedirs(dpath)
            fig.savefig(filepath, dpi=600)
            plt.close()
        else:
            plt.show()
        

    def showLoss(self, xlim=[], ylim=[]):
        print('loss AB')
        self.showLossAB(xlim, ylim)
        print('loss BA')
        self.showLossBA(xlim, ylim)


    def showLossAB(self, xlim=[], ylim=[]):
        g = np.array(self.g_losses)
        g_loss = g[:, 0]
        g_adv = g[:, 1]
        g_recon = g[:, 3]
        g_id = g[:, 5]
        CycleGAN.plot_history(
            [g_loss, g_adv, g_recon, g_id],
            ['g_loss', 'AB discrim', 'AB cycle', 'AB id'],
            xlim,
            ylim)

    def showLossBA(self, xlim=[], ylim=[]):
        g = np.array(self.g_losses)
        g_loss = g[:, 0]
        g_adv = g[:, 2]
        g_recon = g[:, 4]
        g_id = g[:, 6]
        CycleGAN.plot_history(
            [g_loss, g_adv, g_recon, g_id],
            ['g_loss', 'BA discrim', 'BA cycle', 'BA id'],
            xlim,
            ylim)


    @staticmethod
    def plot_history(vals, labels, xlim=[], ylim=[]):
        colors = ['red', 'blue', 'green', 'orange', 'black', 'pink']
        n = len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
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
