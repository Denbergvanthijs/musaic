from collections import Counter

import numpy as np
from smt22.encoders import build_model
from smt22.utils import plots, preprocess, valid_input
from tensorflow.keras.utils import plot_model
from v9.Data.DataGeneratorsTransformer import CombinedGenerator

if __name__ == "__main__":
    # Inputs
    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"
    fp_output = "./src/main/python/smt22/"  # "../../Data/music21

    # Generate data
    combined_generator = CombinedGenerator(fp_music, save_conversion_params=fp_output, to_list=False, meta_prep_f=None)

    # Counter of num_pieces
    num_pieces = combined_generator.get_num_pieces()
    for k, v in Counter(num_pieces).items():
        print(f"{v} songs with {k} instruments")

    # Product of num_pieces
    print(f"Total number of tracks: {sum([k * v for k, v in Counter(num_pieces).items()])}")

    data_iter = combined_generator.generate_data(rhythm_context_size=4, melody_context_size=4)

    cnt = 0
    ys_rhythm, ys_melody = [], []
    X1, X2, X3, X4, X5 = [], [], [], [], []
    for c, (X, y) in enumerate(data_iter):
        X, y_rhythm, y_melody = preprocess(X, y)  # Preprocess a single track

        if not valid_input(X, y_rhythm, y_melody):
            cnt += 1
            continue

        context_rhythms, context_melodies, meta, lead_rhythm, lead_melody = X
        for row in range(context_rhythms.shape[0]):
            X1.append(context_rhythms[row])
            X2.append(context_melodies[row])
            X3.append(meta[row])
            X4.append(lead_rhythm[row])
            X5.append(lead_melody[row])

            ys_rhythm.append(y_rhythm[row])
            ys_melody.append(y_melody[row])

        # if c == 4:  # Early stop for testing
        #     break

    Xs = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5)]
    ys = [np.array(ys_rhythm), np.array(ys_melody)]

    print(f"Xs: {len(Xs)} tracks; {Xs[0].shape} {Xs[1].shape} {Xs[2].shape} {Xs[3].shape} {Xs[4].shape}")
    print(f"Max values: {np.max(Xs[0])} {np.max(Xs[1])} {np.max(Xs[2])} {np.max(Xs[3])} {np.max(Xs[4])}")
    print(f"Min values: {np.min(Xs[0])} {np.min(Xs[1])} {np.min(Xs[2])} {np.min(Xs[3])} {np.min(Xs[4])}")
    print(f"ys_rhythm: {ys[0].shape}")
    print(f"ys_melody: {ys[1].shape}")
    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    model = build_model(127, 4, 25, 48)  # TODO: Train encoder only once, train two seperate decoders
    plot_model(model, to_file="./src/main/python/smt22/model_thijs.png", show_shapes=True, dpi=300)

    hist_model = model.fit(Xs, ys, epochs=10, verbose=1, batch_size=64, validation_split=0.1,
                           shuffle=True, use_multiprocessing=True, workers=6)

    model.save("./src/main/python/smt22/model.h5")
    # model = tf.keras.models.load_model("./src/main/python/smt22/model.h5")

    score = model.evaluate(Xs, ys, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)
    print(f"Rhythm model loss: {score[1]:.4f}; Melody model loss: {score[2]:.4f}")
    print(f"Rhythm model accuracy: {score[3]:.4f}; Melody model accuracy: {score[4]:.4f}")

    plots(hist_model)
