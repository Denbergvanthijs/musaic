import os
from time import asctime

from Data.DataGeneratorsLeadMeta import CombinedGenerator
from Nets.CombinedNetwork import CombinedNetwork
from Nets.MelodyEncoder import MelodyEncoder
from Nets.MelodyNetwork import MelodyNetwork
from Nets.MetaEmbedding import MetaEmbedding
from Nets.MetaPredictor import MetaPredictor
from Nets.RhythmEncoder import BarEmbedding, RhythmEncoder
from Nets.RhythmNetwork import RhythmNetwork
from tensorflow.python.keras.callbacks import TensorBoard

if __name__ == "__main__":
    save_dir = asctime().split()
    save_dir = "_".join([*save_dir[0:3], *save_dir[3].split(":")])
    top_dir = "./src/main/python/v9/Trainings/"

    # Inputs
    fp_input = os.path.join(top_dir, "euroAI_lead")  # Path to saved weights and meta
    fp_meta = os.path.join(fp_input, "meta")  # Path to saved meta
    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"

    # Outputs
    fp_output = os.path.join(top_dir, save_dir)  # Path to save network weights
    fp_logs = os.path.join(fp_output, "logs")  # Path to save logs
    fp_weights = os.path.join(fp_output, "weights")  # Path to save weights

    for fp in [fp_logs, fp_weights]:
        if not os.path.exists(fp):
            os.makedirs(fp)

    # Params
    num_epochs = 200
    checkpoint_frequency = 2
    rhythm_context_size = 4
    melody_context_size = 4

    # Rhythm params
    beat_embed_size = 12
    embed_lstm_size = 24
    out_size = 16

    context_size = rhythm_context_size
    rhythm_enc_lstm_size = 32
    rhythm_dec_lstm_size = 28

    # Melody params
    m = 48
    conv_f = 4
    conv_win_size = 3
    melody_enc_lstm_size = 52
    melody_dec_lstm_size = 32

    meta_data_len = 10

    # Meta
    meta_embedder = MetaEmbedding.from_saved_custom(fp_meta)
    meta_embed_size = meta_embedder.embed_size
    meta_predictor = MetaPredictor.from_saved_custom(fp_meta)
    meta_predictor.freeze()

    # Change
    combined_generator = CombinedGenerator(fp_music, save_conversion_params=fp_output, to_list=0, meta_prep_f=meta_embedder.predict)
    combined_generator.get_num_pieces()

    data_iter = combined_generator.generate_forever(rhythm_context_size=rhythm_context_size,
                                                    melody_context_size=melody_context_size,
                                                    with_metaData=True)
    print("\nData generator set up...\n")
    V_rhythm = combined_generator.rhythm_V
    V_melody = combined_generator.melody_V

    # Individual nets
    bar_embedder = BarEmbedding(V=V_rhythm, beat_embed_size=beat_embed_size, embed_lstm_size=embed_lstm_size, out_size=out_size)
    rhythm_encoder = RhythmEncoder(bar_embedder=bar_embedder, context_size=rhythm_context_size, lstm_size=rhythm_enc_lstm_size)
    rhythm_network = RhythmNetwork(rhythm_encoder=rhythm_encoder, dec_lstm_size=rhythm_dec_lstm_size,
                                   V=V_rhythm, dec_use_meta=True, compile_now=True)

    # Attention: conv_win_size must not be greater than context size!
    conv_win_size = min(melody_context_size, conv_win_size)
    melody_encoder = MelodyEncoder(m=m, conv_f=conv_f, conv_win_size=conv_win_size, enc_lstm_size=melody_enc_lstm_size)
    melody_network = MelodyNetwork(melody_encoder=melody_encoder, rhythm_embed_size=out_size, dec_lstm_size=melody_dec_lstm_size, V=V_melody,
                                   dec_use_meta=True, compile_now=True)

    print("Individual networks set up...\n")

    # Combined network
    comb_net = CombinedNetwork(context_size, m, meta_embed_size, bar_embedder, rhythm_network, melody_network, meta_predictor,
                               generation=False, compile_now=True)

    print("Combined network set up...\n")

    tb = TensorBoard(log_dir=fp_logs)

    # Training loop
    for cur_iteration in range(int(num_epochs / checkpoint_frequency)):
        print("\nITERATION ", cur_iteration)

        epochs = cur_iteration * checkpoint_frequency + checkpoint_frequency
        initial_epoch = cur_iteration * checkpoint_frequency

        comb_net.fit_generator(data_iter, steps_per_epoch=combined_generator.num_pieces,
                               epochs=epochs, initial_epoch=initial_epoch, verbose=2, callbacks=[tb])

        cur_folder_name = os.path.join(fp_weights, "/_checkpoint_" + str(cur_iteration))
        os.makedirs(cur_folder_name)
        comb_net.save_model_custom(cur_folder_name)

    comb_net.save_model_custom(fp_weights)
