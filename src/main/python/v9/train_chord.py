import os
from time import asctime

from Data.DataGeneratorsLeadMetaChords import ChordGenerator
from Nets.ChordNetwork import ChordNetwork
from Nets.CombinedNetwork import CombinedNetwork
from Nets.MelodyEncoder import MelodyEncoder
from Nets.MetaEmbedding import MetaEmbedding
from Nets.MetaPredictor import MetaPredictor


def get_smaller_weights(bigger_melody_encoder, conv_win_size: int) -> list:
    encoder_weights = bigger_melody_encoder.get_weights()
    return [encoder_weights[0][0:conv_win_size]] + encoder_weights[1:]


if __name__ == "__main__":
    save_dir = asctime().split()
    save_dir = "_".join([*save_dir[0:3], *save_dir[3].split(":")])

    fp_input = "./src/main/python/v9/Trainings/euroAI_lead"  # Path to saved weights and meta
    fp_output = os.path.join("./src/main/python/v9/Trainings/", save_dir, "chord")  # Path to save chord weight

    fp_meta = os.path.join(fp_input, "meta")
    fp_weights = os.path.join(fp_input, "weights")

    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"

    size_1 = 1

    for fp in (fp_meta, fp_output, fp_weights):  # Not needed fp_save_dir since parent of others
        if not os.path.exists(fp):
            os.makedirs(fp)

    # Meta
    meta_embedder = MetaEmbedding.from_saved_custom(fp_meta)
    meta_predictor = MetaPredictor.from_saved_custom(fp_meta)
    meta_predictor.freeze()

    chord_generator = ChordGenerator(fp_music, save_conversion_params=fp_output, to_list=False, meta_prep_f=None)
    # data_iter = chord_generator.generate_forever(batch_size=24)
    x, y = chord_generator.list_data()

    combined_network = CombinedNetwork.from_saved_custom(fp_weights, meta_predictor, generation=True, compile_now=False)
    melody_encoder = combined_network.melody_encoder

    melody_encoder_fresh = MelodyEncoder(m=48, conv_f=4, conv_win_size=size_1, enc_lstm_size=52, compile_now=False)
    melody_encoder_fresh.set_weights(get_smaller_weights(melody_encoder, conv_win_size=size_1))

    # for l in melody_encoder_fresh.layers:
    #     l.trainable = False
    melody_encoder_fresh.compile_default()

    chord_network = ChordNetwork(melody_encoder_fresh, 28, chord_generator.V, compile_now=True)
    chord_network.fit(x=x, y=y, epochs=250, verbose=2)

    # Number of chords in bar and number of note values above 12 don't match
    # chord_network.fit_generator(data_iter, steps_per_epoch=50, epochs=1)

    chord_network.save_model_custom(fp_output, save_melody_encoder=True)
