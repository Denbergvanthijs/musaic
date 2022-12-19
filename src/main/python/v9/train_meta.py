import os
from time import asctime

import numpy as np
import numpy.random as rand
from Data.DataGeneratorsLead import CombinedGenerator
from Nets.MetaEmbedding import get_meta_embedder
from Nets.MetaPredictor import MetaPredictor, dirichlet_noise


def gen_meta(comb_gen_instance):
    data_generator = comb_gen_instance.generate_data()
    i = 0
    while True:
        try:
            x, y = next(data_generator)
            yield x[3]
        except IndexError:
            i += 1
            print("IndexError at ", i)
            continue
        except StopIteration:
            return


def gen_preds_and_meta(comb_gen_instance, meta_embedder, forever=False):
    data_generator = comb_gen_instance.generate_data()
    null = meta_embedder.predict(np.zeros((1, 10)))

    while True:
        try:
            x, y = next(data_generator)
            meta = x[3]
            rs_one_hot, ms_one_hot = y

            rand_alph = rand.randint(1, 10)

            def cur_alphs(v):
                return (v * rand_alph) + 1

            rs_noisy = np.asarray([[dirichlet_noise(r_cat, cur_alphs) for r_cat in bar] for bar in rs_one_hot])
            ms_noisy = np.asarray([[dirichlet_noise(m_cat, cur_alphs) for m_cat in bar] for bar in ms_one_hot])

            embedded = meta_embedder.predict(meta)
            padded = np.vstack([null, embedded[:-1]])

            yield [rs_noisy, ms_noisy, padded], embedded

        except IndexError:
            continue

        except StopIteration:
            if not forever:
                return

            data_generator = comb_gen_instance.generate_data()


if __name__ == "__main__":
    save_dir = asctime().split()
    save_dir = "_".join([*save_dir[0:3], *save_dir[3].split(":")])

    fp_meta = os.path.join("./src/main/python/v9/Trainings", save_dir, "meta")
    fp_combined_generator = "./src/main/python/v9/Data/lessfiles"

    if not os.path.exists(fp_meta):
        os.makedirs(fp_meta)

    combined_generator = CombinedGenerator(fp_combined_generator, save_conversion_params=False, to_list=False)
    _ = combined_generator.get_num_pieces()

    meta_examples = rand.permutation(np.vstack(list(gen_meta(combined_generator))))
    meta_embedder, eval_results = get_meta_embedder(meta_examples, embed_size=9, epochs=30, evaluate=True, verbose=1)

    print(f"MetaEmbedding trained!\n\tevaluation results:\n\t{eval_results}")

    pred_meta_gen = gen_preds_and_meta(combined_generator, meta_embedder, forever=True)

    params_rhythm = (None, combined_generator.rhythm_V)
    params_melody = (48, combined_generator.melody_V)

    meta_predictor = MetaPredictor(params_rhythm, params_melody, meta_embedder.embed_size, 8, 12)
    meta_predictor.fit_generator(pred_meta_gen, steps_per_epoch=combined_generator.num_pieces, epochs=4)

    meta_embedder.save_model_custom(fp_meta)
    meta_predictor.save_model_custom(fp_meta)
