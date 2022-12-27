import pickle

import numpy as np
from tensorflow.keras.utils import to_categorical
from v9.Data.DataGeneratorsLeadMeta import DataGenerator as DataGeneratorBase
from v9.Data.DataGeneratorsLeadMeta import MelodyGenerator
from v9.Data.utils import label


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

        print(f"CONVERSION PARAMS SAVED TO {filename}")


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
