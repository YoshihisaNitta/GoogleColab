import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import pickle as pkl
import datetime

import music21
import itertools

###################################################################################
# DataLoader
###################################################################################
class ScoreDataset(tf.keras.utils.Sequence):
    def __init__(self, save_path='music_param.pkl', midi_paths=None, seq_len=32):
        if not midi_paths is None:
            self._build(midi_paths, seq_len)
            self.save(save_path)
        else:
            self.load(save_path)

        
    def _build(self, midi_paths, seq_len=32):
        self.seq_len = seq_len
        
        self.notes_list, self.durations_list = ScoreDataset.makeMusicData(midi_paths)

        notes_set = sorted(set(itertools.chain.from_iterable(self.notes_list))) # flatten 2D -> 1D, Unique, Sort
        durations_set = sorted(set(itertools.chain.from_iterable(self.durations_list)))

        self.note_to_index, self.index_to_note = ScoreDataset.createLookups(notes_set)
        self.duration_to_index, self.index_to_duration = ScoreDataset.createLookups(durations_set)

        self.c_notes = len(self.note_to_index)
        self.c_durations = len(self.duration_to_index)

        self.notes_index_list = ScoreDataset.convertIndex(self.notes_list, self.note_to_index)
        self.durations_index_list = ScoreDataset.convertIndex(self.durations_list, self.duration_to_index)

        self.n_music = len(self.notes_list)
        self.index = 0

        self._build_tbl()       
        

    @staticmethod
    def extractMidi(midi_path):
        notes, durations = [], []
        score = music21.converter.parse(midi_path).chordify()
        for element in score.flat:
            if isinstance(element, music21.note.Note): # Note
                if element.isRest:  # Rest
                    notes.append(str(element.name))   # no pitch, then name only
                else: # note with pitch
                    notes.append(str(element.nameWithOctave))
                durations.append(element.duration.quaterLength) # 1/4 unit
        
            if isinstance(element, music21.chord.Chord): # chord contains multiple notes
                notes.append('.'.join(n.nameWithOctave for n in element.pitches)) # connect with '.'
                durations.append(element.duration.quarterLength) # 1/4 unit
            
        return notes, durations

    
    @staticmethod
    # [notes1, ..., notesN], [durations1, ..., durationsN]
    def makeMusicData(midi_paths):
        notes_list, durations_list = [], []
        for path in midi_paths:
            notes, durations = ScoreDataset.extractMidi(path)  # notes, durations
            notes_list.append(notes)
            durations_list.append(durations)
    
        return notes_list, durations_list


    @staticmethod
    def createLookups(names):  # Lookup Table
        element_to_index = dict((element, idx) for idx, element in enumerate(names))
        index_to_element = dict((idx, element) for idx, element in enumerate(names))
        return element_to_index, index_to_element


    @staticmethod
    def convertIndex(data, element_to_index):
        return [ [ element_to_index[element] for element in x] for x in data]

    
    def getMidiStream(self, g_notes, g_durations):  # [note_index, ...], [duration_index, ...]
        midi_stream = music21.stream.Stream()
        for note_idx, duration_idx in zip(g_notes, g_durations):
            note = self.index_to_note[note_idx]
            duration = self.index_to_duration[duration_idx]
            if ('.' in note): # chord
                notes_in_chord = note.split('.')
                chord_notes = []
                for n_ in notes_in_chord:
                    new_note = music21.note.Note(n_)
                    new_note.duration = music21.duration.Duration(duration)
                    new_note.storeInstrument = music21.instrument.Violoncello()
                    chord_notes.append(new_note)
                new_chord = music21.chord.Chord(chord_notes)
                midi_stream.append(new_chord)
            elif note == 'rest':
                new_note = music21.note.Rest()
                new_note.duration = music21.duration.Duration(duration)
                new_note.storedInstrument = music21.instrument.Violoncello()
                midi_stream.append(new_note)
            else:
                new_note = music21.note.Note(note)
                new_note.duration = music21.duration.Duration(duration)
                new_note.storedInstrument = music21.instrument.Violoncello()
                midi_stream.append(new_note)

        return midi_stream
        
        
    def _build_tbl(self):
        a = [ len(x)-self.seq_len for x in self.notes_list ]  # [int, int, ...]
        for i in range(1, len(a)):   # cumulative frequency of data
            a[i] = a[i-1] + a[i]
        self.cumulative_freq = a
        #print(f'cumulative_freq: {self.cumulative_freq}')

            
    def searchTbl(self, index):
        index = index % self.__len__()
        low = 0
        high = self.n_music - 1
        for i in range(self.n_music):
            mid = (low + high) // 2
            #print(f'{i}/{self.n_music}: {high} {low} {mid} {index}')
            if self.cumulative_freq[mid] > index:
                if mid == 0 or self.cumulative_freq[mid-1] <= index:
                    return mid
                high = mid - 1
            else:
                low = mid + 1

                
    def __len__(self):
        return self.cumulative_freq[-1]

    
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
            return self.__getitemList__(range(start, stop, step))
        
        elif isinstance(index, int):
            return self.__getitemInt__(index)
        
        else:
            return self.__getitemList__(index)
        
        
    def __getitemList__(self, indices):
        x_notes, x_durations, y_notes, y_durations = [], [], [], []
        for i in indices:
            [x_note, x_duration], [y_note, y_duration] = self.__getitemInt__(i)
            x_notes.append(x_note)
            x_durations.append(x_duration)
            y_notes.append(y_note)
            y_durations.append(y_duration)

        return (x_notes, x_durations), (y_notes, y_durations)
        

        
    def __getitemInt__(self, index):
        index = index % self.__len__()
        #print(f'index = {index} {self.__len__()}')
        tbl_idx = self.searchTbl(index)
        #print(f'tbl_idx = {tbl_idx}')
        tgt = index
        if (tbl_idx > 0):
            tgt -= self.cumulative_freq[tbl_idx - 1]
        #print(f'tgt = {tgt}')
        
        x_note = self.notes_index_list[tbl_idx][tgt: (tgt + self.seq_len)]
        y_note = self.notes_index_list[tbl_idx][tgt + self.seq_len]
        x_duration = self.durations_index_list[tbl_idx][tgt: (tgt + self.seq_len)]
        y_duration = self.durations_index_list[tbl_idx][tgt + self.seq_len]
        
        #ohv_y_note = tf.keras.utils.to_categorical(y_note, self.c_notes)
        #ohv_y_duration = tf.keras.utils.to_categorical(y_duration, self.c_durations)
        
        return (x_note, x_duration), (y_note, y_duration)

    
    def __next__(self):
        self.index += 1
        return self.__getitem__(self.index-1)


    def save(self, filepath):
        dpath, fname = os.path.split(filepath)
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        with open(filepath, 'wb') as f:
            pkl.dump([
                self.seq_len,
                self.notes_list,
                self.durations_list,
                self.note_to_index,
                self.index_to_note,
                self.duration_to_index,
                self.index_to_duration,
                self.c_notes,
                self.c_durations,
                self.notes_index_list,
                self.durations_index_list,
                self.n_music,
                self.index,
                self.cumulative_freq
            ], f)
    

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            params = pkl.load(f)
            
        [
            self.seq_len,
            self.notes_list,
            self.durations_list,
            self.note_to_index,
            self.index_to_note,
            self.duration_to_index,
            self.index_to_duration,
            self.c_notes,
            self.c_durations,
            self.notes_index_list,
            self.durations_index_list,
            self.n_music,
            self.index,
            self.cumulative_freq
        ] = params
    

###################################################################################
# Model
###################################################################################

class LSTMMusic():
    def __init__(self,
                 c_notes,
                 c_durations,
                 seq_len = 32,
                 optimizer='adam',
                 learning_rate = 0.001,
                 embed_size = 100,
                 rnn_units = 256,
                 use_attention = True,
                 epochs = 0,
                 losses = [],
                 n_losses = [],
                 d_losses = [],
                 val_losses = [],
                 val_n_losses = [],
                 val_d_losses = []
                 ):
        self.c_notes = c_notes
        self.c_durations = c_durations
        self.seq_len = seq_len
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.embed_size = embed_size
        self.rnn_units = rnn_units
        self.use_attention = use_attention
        self.epochs = epochs
        self.losses = losses
        self.n_losses = n_losses
        self.d_losses = d_losses
        self.val_losses = val_losses
        self.val_n_losses = val_n_losses
        self.val_d_losses = val_d_losses

        self.model, self.att_model = self._create_network(c_notes, c_durations, embed_size, rnn_units, use_attention)
        self.cce1 = tf.keras.losses.CategoricalCrossentropy(),
        self.cce2 = tf.keras.losses.CategoricalCrossentropy(),

        
    def _create_network(
            self,
            n_notes, 
            n_durations, 
            embed_size, 
            rnn_units, 
            use_attention
    ):
        notes_in = tf.keras.layers.Input(shape=(None,))
        durations_in = tf.keras.layers.Input(shape=(None,))
    
        x1 = tf.keras.layers.Embedding(n_notes, embed_size)(notes_in)
        x2 = tf.keras.layers.Embedding(n_durations, embed_size)(durations_in)
    
        x = tf.keras.layers.Concatenate()([x1, x2])
    
        x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
        # x = tf.keras.layers.Dropout(0.2)(x)
        
        if use_attention:
            x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
            # x = tf.keras.layers.Dropout(0.2)(x)
            
            e = tf.keras.layers.Dense(1, activation='tanh')(x)
            e = tf.keras.layers.Reshape([-1])(e)   # batch_size * N 
            alpha = tf.keras.layers.Activation('softmax')(e)
            
            alpha_repeated = tf.keras.layers.Permute([2,1])(tf.keras.layers.RepeatVector(rnn_units)(alpha))
        
            c = tf.keras.layers.Multiply()([x, alpha_repeated])
            c = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1), output_shape=(rnn_units,))(c)
        
        else:
        
            c = tf.keras.layers.LSTM(rnn_units)(x)
            #c = tf.keras.layers.Dropout(0.2)(c)
    
        notes_out = tf.keras.layers.Dense(n_notes, activation='softmax', name='pitch')(c)
        durations_out = tf.keras.layers.Dense(n_durations, activation='softmax', name='duration')(c)
    
        model = tf.keras.models.Model([notes_in, durations_in], [notes_out, durations_out])
    
        if use_attention:
            att_model = tf.keras.models.Model([notes_in, durations_in], alpha)
        else:
            att_model = None
            
        return model, att_model


    def get_opti(self, learning_rate):
        if self.optimizer == 'adam':
            opti = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1 = 0.5)
        elif self.optimizer == 'rmsprop':
            opti = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opti = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return opti


    def _compile(self):
        opti = self.get_opti(self.learning_rate)
        self.model.compile(
            loss=[
                self.cce1,
                self.cce2
            ],
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = self.learning_rate)
        )

        
    def train_with_fit(
            self,
            xs,   # (x_notes,x_durations)
            ys,   # (y_notes, y_durations)
            epochs=1,
            batch_size=32,
            run_folder='run/',
            shuffle=False,
            validation_data = None
    ):
        history = self.model.fit(
            xs,
            ys,
            initial_epoch = self.epochs,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = shuffle,
            validation_data = validation_data
        )
        self.epochs = epochs

        h = history.history
        self.losses += h['loss']
        self.n_losses += h['pitch_loss']
        self.d_losses += h['duration_loss']
        
        if not validation_data is None:
            self.val_losses += h['val_loss']
            self.val_n_losses += h['val_pitch_loss']
            self.val_d_losses += h['val_duration_loss']
            
        self.save(run_folder)
        self.save(run_folder, self.epochs)

        if validation_data is None:
            return self.losses, self.n_losses, self.d_losses
        else:
            return self.losses, self.n_losses, self.d_losses, self.val_losses, self.val_n_losses, self.val_d_losses,


    @tf.function
    def loss_fn(self, y_notes, y_durations, p_notes, p_durations):
        #print(y_notes.shape, p_notes.shape, y_durations.shape, p_durations.shape)
        #y_notes = np.array(y_notes, dtype='float32')
        #p_notes = np.array(p_notes, dtype='float32')
        n_loss = tf.keras.losses.CategoricalCrossentropy()(y_notes, p_notes)
        d_loss = tf.keras.losses.CategoricalCrossentropy()(y_durations, p_durations)
        #n_loss = self.cce1(y_notes, p_notes)
        #d_loss = self.cce2(y_durations, p_durations)
        loss = tf.add(n_loss, d_loss)
        return loss, n_loss, d_loss


    @tf.function
    def train_step(
            self,
            x_notes, 
            x_durations, 
            y_notes, 
            y_durations, 
            optimizer
    ):
        with tf.GradientTape() as tape:
            p_notes, p_durations = self.model([x_notes, x_durations])
            loss, note_loss, duration_loss = self.loss_fn(y_notes, y_durations, p_notes, p_durations)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, note_loss, duration_loss

    
    def train(
            self,
            xs,
            ys,
            epochs=1, 
            batch_size=32, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            shuffle=False,
            run_folder='run/',
            print_step_interval = 100,
            save_epoch_interval = 100,
            validation_data = None
    ):
        start_time = datetime.datetime.now()

        x_notes_all, x_durations_all = xs
        y_notes_all, y_durations_all = ys

        steps = len(x_notes_all) // batch_size

        for epoch in range(self.epochs, epochs):
            indices = tf.range(len(x_notes_all), dtype=tf.int32)
            if shuffle:
                indices = tf.random.shuffle(indices)

            step_losses, step_n_losses, step_d_losses = [], [], []
            
            for step in range(steps):
                start = batch_size * step
                end = start + batch_size
                
                idxs = indices[start:end]
                x_notes = x_notes_all[idxs]
                x_durations = x_durations_all[idxs]
                y_notes = np.array(y_notes_all[idxs])
                y_durations = np.array(y_durations_all[idxs])
                
                n_ = np.array(x_notes, dtype='float32')
                d_ = np.array(x_durations, dtype='float32')
                         
                batch_loss, batch_n_loss, batch_d_loss = self.train_step(
                    n_, 
                    d_, 
                    y_notes, 
                    y_durations,
                    optimizer
                )

                step_losses.append(np.mean(batch_loss))
                step_n_losses.append(np.mean(batch_n_loss))
                step_d_losses.append(np.mean(batch_d_loss))
                
                elapsed_time = datetime.datetime.now() - start_time
                if (step+1) % print_step_interval == 0:
                    print(f'{epoch+1}/{epochs} {step+1}/{steps} loss {step_losses[-1]:.3f} pitch_loss {step_n_losses[-1]:.3f} duration_loss {step_d_losses[-1]:.3f} {elapsed_time}')


            epoch_loss = np.mean(step_losses)
            epoch_n_loss = np.mean(step_n_losses)
            epoch_d_loss = np.mean(step_d_losses)

            self.losses.append(epoch_loss)
            self.n_losses.append(epoch_n_loss)
            self.d_losses.append(epoch_d_loss)

            val_str = ''
            if not validation_data is None:
                (val_x_notes, val_x_durations), (val_y_notes, val_y_durations) = validation_data
                p_notes, p_durations = self.model([val_x_notes, val_x_durations])
                val_loss, val_n_loss, val_d_loss = self.loss_fn(
                    val_y_notes,
                    val_y_durations,
                    p_notes,
                    p_durations
                )
                val_loss = np.mean(val_loss)
                val_n_loss = np.mean(val_n_loss)
                val_d_loss = np.mean(val_d_loss)

                self.val_losses.append(val_loss)
                self.val_n_losses.append(val_n_loss)
                self.val_d_losses.append(val_d_loss)

                val_str = f'val_loss {val_loss:.3f} val_pitch_loss {val_n_loss:.3f} val_duration_loss {val_d_loss:.3f}'
                
            self.epochs += 1

            elapsed_time = datetime.datetime.now() - start_time
            print(f'{self.epochs}/{epochs} loss {epoch_loss:.3f} pitch_loss {epoch_n_loss:.3f} duration_loss {epoch_d_loss:.3f} {val_str} {elapsed_time}')
            
            if self.epochs % save_epoch_interval == 0:
                self.save(run_folder)
                self.save(run_folder, self.epochs)

        self.save(run_folder)
        self.save(run_folder, self.epochs)
        
        return self.losses, self.n_losses, self.d_losses, self.val_losses, self.val_n_losses, self.val_d_losses


    @staticmethod
    def sample_with_temperature(preds, temperature):
        if temperature == 0:
            return np.argmax(preds)
        else:
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            return np.random.choice(len(preds), p=preds)
        

    def generate(self, s_notes, s_durations, count=64, note_temperature = 0.5, duration_temperature = 0.5):
        x_notes = np.array([s_notes], dtype='float32')
        x_durations = np.array([s_durations], dtype='float32')
            
        g_notes, g_durations = [], []
        for i in range(count):
            p_notes, p_durations = self.model([x_notes, x_durations])
            note = LSTMMusic.sample_with_temperature(p_notes[0], note_temperature)
            duration = LSTMMusic.sample_with_temperature(p_durations[0], duration_temperature)
            g_notes.append(note)
            g_durations.append(duration)

            x_notes = np.roll(x_notes, -1)
            x_durations = np.roll(x_durations, -1)
            x_notes[0,-1] = note
            x_durations[0,-1] = duration

        return g_notes, g_durations  # [note_index, ...], [duration_index, ...]


    def save(self, folder, epoch=None):
        self.save_params(folder, epoch)
        self.save_weights(folder, epoch)


    @staticmethod
    def load(folder, epoch=None):
        params = LSTMMusic.load_params(folder, epoch)
        music = LSTMMusic(*params)
        music.load_weights(folder, epoch)
        return music


    def save_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.save_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))    
            self.save_model_weights(self.att_model, os.path.join(run_folder, 'weights/weights_att.h5'))    
        else:
            self.save_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))
            self.save_model_weights(self.att_model, os.path.join(run_folder, f'weights/weights_att_{epoch}.h5'))


    def load_weights(self, run_folder, epoch=None):
        if epoch is None:
            self.load_model_weights(self.model, os.path.join(run_folder, 'weights/weights.h5'))
            self.load_model_weights(self.att_model, os.path.join(run_folder, 'weights/weights_att.h5'))
        else:
            self.load_model_weights(self.model, os.path.join(run_folder, f'weights/weights_{epoch}.h5'))
            self.load_model_weights(self.att_model, os.path.join(run_folder, f'weights/weights_att_{epoch}.h5'))


    def save_model_weights(self, model, filepath):
        dpath, fname = os.path.split(filepath)
        if dpath != '' and not os.path.exists(dpath):
            os.makedirs(dpath)
        model.save_weights(filepath)


    def load_model_weights(self, model, filepath):
        model.load_weights(filepath)


    def save_params(self, folder, epoch=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        if epoch is None:
            filepath = os.path.join(folder, 'params.pkl')
        else:
            filepath = os.path.join(folder, f'params_{epoch}.pkl')

        with open(filepath, 'wb') as f:
            pkl.dump([
                self.c_notes,
                self.c_durations,
                self.seq_len,
                self.optimizer,
                self.learning_rate,
                self.embed_size,
                self.rnn_units,
                self.use_attention,
                self.epochs,
                self.losses,
                self.n_losses,
                self.d_losses,
                self.val_losses,
                self.val_n_losses,
                self.val_d_losses
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


    @staticmethod
    def plot_history(vals, labels):
        colors = ['red', 'blue', 'green', 'black', 'orange', 'pink', 'purple', 'olive', 'cyan']
        n = len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        for i in range(n):
            ax.plot(vals[i], c=colors[i], label=labels[i])
        ax.legend(loc='upper right')
        ax.set_xlabel('epochs')

        plt.show()

        
