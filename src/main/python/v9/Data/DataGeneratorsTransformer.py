from itertools import tee

import numpy as np
from Data.DataGeneratorsLeadMeta import \
    CombinedGenerator as CombinedGeneratorBase
from Data.DataGeneratorsLeadMeta import MelodyGenerator
from Data.DataGeneratorsLeadMeta import RhythmGenerator as RhythmGeneratorBase
from keras.utils import to_categorical


class RhythmGenerator(RhythmGeneratorBase):
    def __init__(self, path, save_conversion_params=True, to_list=False, meta_prep_f=None):
        super().__init__(path, save_conversion_params, to_list, meta_prep_f)

    def generate_data(self, context_size=1, rand_stream=None, with_rhythms=False, with_metaData=True):
        song_iter = self.get_rhythms_together(with_metaData=with_metaData)

        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")

        for instrument_ls in song_iter:
            cur_i = next(rand_stream)

            cur_lead, _ = instrument_ls[cur_i]
            lead_labeled, _ = self.prepare_piece(cur_lead, context_size)

            for rhythms, meta in instrument_ls:
                rhythms_labeled, context_ls = self.prepare_piece(rhythms, context_size)

                if with_metaData:
                    prepared_meta = np.array(list(map(self.prepare_metaData, meta)))
                    context_ls.append(prepared_meta)

                context_ls.append(lead_labeled)

                yield context_ls, to_categorical(rhythms_labeled, num_classes=self.V)


class CombinedGenerator(CombinedGeneratorBase):
    def __init__(self, path, save_conversion_params=True, to_list=False, meta_prep_f=None):
        super().__init__(path, save_conversion_params=save_conversion_params, to_list=to_list, meta_prep_f=meta_prep_f)

        self.rhythm_gen = RhythmGenerator(path, save_conversion_params=save_conversion_params, to_list=to_list, meta_prep_f=meta_prep_f)
        self.melody_gen = MelodyGenerator(path, save_conversion_params=save_conversion_params, to_list=to_list, meta_prep_f=meta_prep_f)

        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V

    def generate_data(self, rhythm_context_size=1, melody_context_size=1, random_stream=None, with_metaData=False):
        if not random_stream:
            random_stream = self.random_stream()

        r1, r2 = tee(random_stream, 2)

        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size, rand_stream=r1, with_rhythms=False, with_metaData=True)
        melody_iter = self.melody_gen.generate_data(melody_context_size, rand_stream=r2, with_metaData=False)

        for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
            rhythm_x, rhythm_y = cur_rhythm
            (*context_rhythms, meta, rhythm_lead) = rhythm_x  # Four values for context, one for meta, one for lead

            (melody_x, melody_lead), melody_y = cur_melody
            melody_lead = melody_lead.reshape((-1, 1, 48))

            X = [*context_rhythms, melody_x, meta, rhythm_lead, melody_lead]
            y = [rhythm_y, melody_y]

            yield X, y
