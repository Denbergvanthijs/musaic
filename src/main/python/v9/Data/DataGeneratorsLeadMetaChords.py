import pickle
import random
from itertools import tee

import numpy as np
from Data.DataGeneratorsLeadMeta import DataGenerator as DataGeneratorBase
from Data.utils import label
from keras.utils import to_categorical


class DataGenerator(DataGeneratorBase):
    def __init__(self, path, save_conversion_params=None, to_list=False, meta_prep_f=None):
        super().__init__(path, save_conversion_params, to_list, meta_prep_f)

    def save_conversion_params(self, filename=None):
        if not self.conversion_params:
            raise ValueError("DataGenerator.save_conversion_params called while DataGenerator.conversion_params is empty.")

        if not filename:
            filename = self.save_dir + "/" + "DataGenerator.conversion_params"

        with open(filename, "wb") as handle:
            pickle.dump(self.conversion_params, handle)

        print("CONVERSION PARAMS SAVED TO " + filename)


class RhythmGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True,
                 to_list=False, meta_prep_f=None):
        super().__init__(path,
                         save_conversion_params=save_conversion_params,
                         to_list=to_list, meta_prep_f=meta_prep_f)
        song_iter = self.get_rhythms_together(with_metaData=False)
        label_f, self.label_d = label([beat
                                       for instruments in song_iter
                                       for s in instruments
                                       for bar in s
                                       for beat in bar], start=0)

        self.null_elem = ()
        self.V = len(self.label_d)
        self.conversion_params["rhythm"] = self.label_d
        if self.save_params_eager:
            self.save_conversion_params()

    def get_rhythms_together(self, with_metaData=True):
        yield from self.get_songs_together(lambda d: d.__getitem__("rhythm"),
                                           with_metaData=with_metaData)

    def generate_data(self, context_size=1, rand_stream=None, with_rhythms=True,
                      with_metaData=True):
        song_iter = self.get_rhythms_together(with_metaData=True)

        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")

        for instrument_ls in song_iter:
            cur_i = next(rand_stream)
            # print("rhythm rand ind = ", cur_i)
            cur_lead, _ = instrument_ls[cur_i]
            lead_labeled, _ = self.prepare_piece(cur_lead,
                                                 context_size)

            for rhythms, meta in instrument_ls:
                rhythms_labeled, context_ls = self.prepare_piece(rhythms,
                                                                 context_size)

                if with_rhythms:
                    context_ls.append(rhythms_labeled)

                if with_metaData:
                    prepared_meta = np.array(list(map(self.prepare_metaData, meta)))
                    context_ls.append(prepared_meta)
                    prev_meta = np.vstack([np.zeros_like(prepared_meta[0]),
                                           prepared_meta[:-1]])
                    context_ls.append(prev_meta)

                context_ls.append(lead_labeled)

                yield (context_ls, to_categorical(rhythms_labeled, num_classes=self.V))

    def prepare_piece(self, rhythms, context_size):
        bar_len = len(rhythms[0])
        rhythms_labeled = [tuple(self.label_d[b] for b in bar) for bar in rhythms]
        null_bar = (self.label_d[self.null_elem], ) * bar_len

        padded_rhythms = [null_bar] * context_size + rhythms_labeled
        contexts = [padded_rhythms[i:-(context_size - i)] for i in range(context_size)]
        return np.asarray(rhythms_labeled), list(map(np.asarray, contexts))


class MelodyGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True,
                 to_list=False, meta_prep_f=None):
        super().__init__(path,
                         save_conversion_params=save_conversion_params,
                         to_list=to_list, meta_prep_f=meta_prep_f)

        # song_iter = self.get_notevalues_together(with_metaData=False)
        # self.V = len(set(n for instruments in song_iter
        #                  for melodies in instruments
        #                  for bar in melodies for n in bar))
        self.V = 25
        self.null_elem = 0

    def get_notevalues_together(self, with_metaData=True):
        song_iter = self.get_songs_together(lambda d: d["melody"]["notes"],
                                            with_metaData=with_metaData)

        if with_metaData:
            for instruments in song_iter:
                cur_ls = []
                for melodies, meta in instruments:
                    melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies]
                    cur_ls.append((melodies_None_replaced, meta))
                yield cur_ls
        else:
            for instruments in song_iter:
                cur_ls = []
                for melodies in instruments:
                    melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies]
                    cur_ls.append(melodies_None_replaced)
                yield cur_ls

    def generate_data(self, context_size=1, rand_stream=None, with_metaData=True):
        song_iter = self.get_notevalues_together(with_metaData=True)

        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")

        for instrument_ls in song_iter:
            cur_i = next(rand_stream)
            # print("melody rand ind = ", cur_i)
            cur_lead, _ = instrument_ls[cur_i]
            lead_mat, _ = self.prepare_piece(cur_lead, instrument_ls,
                                             context_size)

            for melodies, meta in instrument_ls:
                melodies_mat, contexts = self.prepare_piece(melodies,
                                                            instrument_ls,
                                                            context_size)

                melodies_y = to_categorical(melodies_mat, num_classes=self.V)
                melodies_y[:, :, 0] = 0.

                if with_metaData:
                    prepared_meta = np.array(list(map(self.prepare_metaData, meta)))
                    prev_meta = np.vstack([np.zeros_like(prepared_meta[0]),
                                           prepared_meta[:-1]])
                    yield ([contexts,
                            prepared_meta,
                            prev_meta,
                            lead_mat],
                           melodies_y)
                else:
                    yield ([contexts, lead_mat],
                           melodies_y)

    def prepare_piece(self, melodies, instrument_ls, context_size):
        bar_len = len(melodies[0])
        null_bar = (self.null_elem, ) * bar_len

        filled_melodies = self.fill_melodies(melodies, instrument_ls)
        melodies_mat = np.asarray(filled_melodies)

        padded_melodies = [null_bar] * context_size + filled_melodies
        contexts = [padded_melodies[i:-(context_size - i)] for i in range(context_size)]
        contexts = np.transpose(np.asarray(contexts), axes=(1, 0, 2))
        return melodies_mat, contexts

    def fill_melodies(self, melodies, instrument_ls):
        filled_melodies = [[n for n in bar] for bar in melodies]

        for i, bar in enumerate(melodies):
            try:
                note_pool = set([n for ins, _ in instrument_ls for n in ins[i] if n > 0])
            except IndexError as e:
                raise IndexError(e.args[0] +
                                 "\nATTENTION: not all instruments have the same number of bars in some songs\n" +
                                 "(causes a fail due to IndexError)")
                # do this to work around the IndexError; will cause if below to trigger
                note_pool = []
            if len(note_pool) == 0:
                note_pool.add(1)
            if len(note_pool) == 1:
                note_pool.add(8)

            for j, note in enumerate(bar):
                if note > 0:
                    note_pool.add(note)
                else:
                    filled_melodies[i][j] = random.sample(note_pool, 1)[0]

        return filled_melodies


class ChordGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=False,
                 to_list=False, meta_prep_f=None):

        super().__init__(path,
                         save_conversion_params=False,
                         to_list=to_list, meta_prep_f=meta_prep_f)

        song_iter = self.get_chords_together(with_metaData=False)

        _, self.label_d = label([chord
                                 for instruments in song_iter
                                 for s in instruments
                                 for bar in s
                                 for chord in bar], start=0)

        self.V = len(self.label_d)
        if save_conversion_params:
            self.conversion_params["chords"] = self.label_d
            self.save_conversion_params(filename=save_conversion_params +
                                        "/ChordGenerator.conversion_params")

        self.melody_gen = MelodyGenerator(path, save_conversion_params=False,
                                          to_list=to_list, meta_prep_f=None)

    def get_chords_together(self, with_metaData=True):
        yield from self.get_songs_together(lambda d: d["melody"]["chords"],
                                           with_metaData=with_metaData)

    def generate_data(self):
        melody_iter = self.melody_gen.get_notevalues_together(with_metaData=True)

        chord_iter = self.get_chords_together(with_metaData=True)

        for ins_chords, ins_melody in zip(chord_iter, melody_iter):
            for (chords, meta), (melodies, meta_mel) in zip(ins_chords, ins_melody):
                melodies_mat, _ = self.melody_gen.prepare_piece(melodies,
                                                                ins_melody,
                                                                context_size=1)
                meta_prepared = np.array(list(map(self.prepare_metaData, meta)))

                for bar_chords, bar_melody, meta_bar in zip(chords, melodies_mat,
                                                            meta_prepared):

                    if bar_chords:
                        chord_notes = [n for n in bar_melody if n > 12]
                        # print([n for n in bar_melody if n])
                        # print(chord_notes)
                        # print(len(chord_notes))
                        # print(bar_chords)
                        # print(len(bar_chords), "\n")
                        for n, single_chord in zip(chord_notes, bar_chords):
                            n_a = np.asarray([n])
                            chord_one_hot = to_categorical([self.label_d[single_chord]],
                                                           num_classes=self.V)
                            yield [n_a, bar_melody.reshape(1, -1), meta_bar],\
                                chord_one_hot.reshape((-1, ))

                            # ! Number of chords in bar and number of note values above 12 don't match !

    def generate_forever(self, batch_size):
        data_gen = self.generate_data()
        while True:
            cur_batch_x, cur_batch_y = [], []
            i = 0
            for x, y in data_gen:
                cur_batch_x.append(x)
                cur_batch_y.append(y)
                if i == batch_size:
                    cur_batch_x = [np.asarray(val_ls) for val_ls in zip(*cur_batch_x)]
                    cur_batch_y = np.vstack(cur_batch_y)
                    yield cur_batch_x, cur_batch_y
                    cur_batch_x, cur_batch_y = [], []
                    i = 0

                i += 1

            data_gen = self.generate_data()

    def list_data(self):
        data_iter = self.generate_data()

        xs, ys = list(zip(*data_iter))
        x = []
        for x_i in zip(*xs):
            x.append(np.asarray(x_i))

        y = np.asarray(ys)

        return x, y


class CombinedGenerator(DataGenerator):
    def __init__(self, path,
                 save_conversion_params=True,
                 to_list=False, meta_prep_f=None):
        super().__init__(path,
                         save_conversion_params=save_conversion_params,
                         to_list=to_list, meta_prep_f=meta_prep_f)
        self.rhythm_gen = RhythmGenerator(path,
                                          save_conversion_params=save_conversion_params,
                                          to_list=to_list,
                                          meta_prep_f=meta_prep_f)
        self.melody_gen = MelodyGenerator(path,
                                          save_conversion_params=save_conversion_params,
                                          to_list=to_list,
                                          meta_prep_f=meta_prep_f)

        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V

    # def random_stream(self):
    #     rhythm_ls = map(len,
    #                     self.rhythm_gen.get_rhythms_together(with_metaData=False))
    #     melody_ls = map(len,
    #                     self.melody_gen.get_notevalues_together(with_metaData=False))

    #     for rl, ml in zip(rhythm_ls, melody_ls):
    #         if not rl == ml:
    #             raise ValueError("CombinedGenerator.random_stream:\n" +
    #                              "number of instruments in rhythm unequal " +
    #                              "number of instruments in melody!")

    #         yield rand.randint(rl)

    def generate_data(self, rhythm_context_size=1, melody_context_size=1,
                      random_stream=None, with_metaData=True):

        if not random_stream:
            random_stream = self.random_stream()

        r1, r2 = tee(random_stream, 2)

        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size,
                                                    rand_stream=r1,
                                                    with_rhythms=True,
                                                    with_metaData=with_metaData)
        melody_iter = self.melody_gen.generate_data(melody_context_size,
                                                    rand_stream=r2,
                                                    with_metaData=False)

        if with_metaData:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, meta, prev_meta, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody
                melody_lead = melody_lead.reshape((-1, 1, 48))
                yield ([*rhythm_x, rhythms, melody_x, meta, prev_meta, rhythm_lead, melody_lead],
                       [rhythm_y, melody_y, meta])
        else:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody
            yield ([*rhythm_x, rhythms, melody_x, rhythm_lead, melody_lead],
                   [rhythm_y, melody_y])


# dg = DataGenerator("../../Data/music21", save_conversion_params=0,
#                   to_list=0, meta_prep_f=None)
# chord_list = []
#
# for s in dg.load_songs():
#    for i in range(s["instruments"]):
#        cur_ins = s[i]
#        melodies = cur_ins["melody"]["notes"]
#        chords = cur_ins["melody"]["chords"]
#        if any(chords):
# print(chords)
# print("\n")
#            for bar_ch, bar_mel, in zip(chords, melodies):
#                print(bar_ch)
#                print(bar_mel)
#                for c in bar_ch:
#                    chord_list.append(c)
#            break
#
#
#
#    print(".", end="")
#
# 9721
# ((0, 4, 7), 2179),
# ((0, 4, -5), 1499),
# ((0, 3, 7), 1441),
# ((0, -8, -5), 1174),
# ((0, 3, -5), 995),
# ((0, 4), 550),
# ((0, 3), 412),
#
# chg = ChordGenerator("../../Data/music21", save_conversion_params=False,
#                     to_list=False, meta_prep_f=None)
#
# ch_iter = chg.generate_data()
#
# ls = [ch for ch, mel, met in ch_iter]
