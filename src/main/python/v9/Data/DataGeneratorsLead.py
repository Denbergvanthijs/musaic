import random
from itertools import tee

import numpy as np
import numpy.random as rand
from Data.DataGenerators import DataGenerator as DataGeneratorBase
from Data.utils import label
from keras.utils import to_categorical


class DataGenerator(DataGeneratorBase):
    def __init__(self, path: str, save_conversion_params: bool = True, to_list: bool = False) -> None:
        super().__init__(path, save_conversion_params, to_list)

    def get_songs_together(self, getitem_function, with_metaData=True):
        for song in self.load_songs():
            num_ins = song["instruments"]
            if with_metaData:
                yield [(getitem_function(song[i]), song[i]["metaData"]) for i in range(num_ins)]
            else:
                yield [getitem_function(song[i]) for i in range(num_ins)]

    def random_stream(self):
        instrument_nums = self.get_num_pieces()

        for n_ins in instrument_nums:
            yield rand.randint(n_ins)


class RhythmGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True, to_list=False):
        super().__init__(path, save_conversion_params=save_conversion_params, to_list=to_list)
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
        yield from self.get_songs_together(lambda d: d.__getitem__("rhythm"), with_metaData=with_metaData)

    def generate_data(self, context_size=1, rand_stream=None, with_rhythms=True, with_metaData=True):
        song_iter = self.get_rhythms_together(with_metaData=True)

        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")

        for instrument_ls in song_iter:
            cur_i = next(rand_stream)
            # print("rhythm rand ind = ", cur_i)
            cur_lead, _ = instrument_ls[cur_i]
            lead_labeled, _ = self.prepare_piece(cur_lead, context_size)

            for rhythms, meta in instrument_ls:
                rhythms_labeled, context_ls = self.prepare_piece(rhythms, context_size)

                if with_rhythms:
                    context_ls.append(rhythms_labeled)

                if with_metaData:
                    prepared_meta = np.array(list(map(self.prepare_metaData, meta)))
                    context_ls.append(prepared_meta)

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
    def __init__(self, path, save_conversion_params=True, to_list=False):
        super().__init__(path, save_conversion_params=save_conversion_params, to_list=to_list)

        # song_iter = self.get_notevalues_together(with_metaData=False)
        # self.V = len(set(n for instruments in song_iter for melodies in instruments for bar in melodies for n in bar))

        self.V = 25
        self.null_elem = 0

    def get_notevalues_together(self, with_metaData=True):
        song_iter = self.get_songs_together(lambda d: d["melody"]["notes"], with_metaData=with_metaData)

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
            lead_mat, _ = self.prepare_piece(cur_lead, instrument_ls, context_size)

            for melodies, meta in instrument_ls:
                melodies_mat, contexts = self.prepare_piece(melodies, instrument_ls, context_size)

                melodies_y = to_categorical(melodies_mat, num_classes=self.V)
                melodies_y[:, :, 0] = 0.

                if with_metaData:
                    prepared_meta = np.array(list(map(self.prepare_metaData, meta)))

                    yield ([contexts, prepared_meta.lead_mat], melodies_y)
                else:
                    yield ([contexts, lead_mat], melodies_y)

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


class CombinedGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True, to_list=False):
        super().__init__(path, save_conversion_params=save_conversion_params, to_list=to_list)
        self.rhythm_gen = RhythmGenerator(path, save_conversion_params=save_conversion_params, to_list=to_list)
        self.melody_gen = MelodyGenerator(path, save_conversion_params=save_conversion_params, to_list=to_list)

        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V

    # def random_stream(self):
    #     rhythm_ls = map(len, self.rhythm_gen.get_rhythms_together(with_metaData=False))
    #     melody_ls = map(len, self.melody_gen.get_notevalues_together(with_metaData=False))

    #     for rl, ml in zip(rhythm_ls, melody_ls):
    #         if not rl == ml:
    #             raise ValueError("CombinedGenerator.random_stream:\n number of instruments in rhythm unequal number of instruments in melody!")

    #         yield rand.randint(rl)

    def generate_data(self, rhythm_context_size=1, melody_context_size=1, random_stream=None, with_metaData=True):
        if not random_stream:
            random_stream = self.random_stream()

        r1, r2 = tee(random_stream, 2)

        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size, rand_stream=r1, with_rhythms=True, with_metaData=with_metaData)
        melody_iter = self.melody_gen.generate_data(melody_context_size, rand_stream=r2, with_metaData=False)

        if with_metaData:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, meta, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody

                yield ([*rhythm_x, rhythms, melody_x, meta, rhythm_lead, melody_lead], [rhythm_y, melody_y])
        else:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody

            yield ([*rhythm_x, rhythms, melody_x, rhythm_lead, melody_lead], [rhythm_y, melody_y])
