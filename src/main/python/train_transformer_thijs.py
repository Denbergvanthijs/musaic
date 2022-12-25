from collections import Counter

import numpy as np
from smt22.encoders import build_model
from smt22.utils import plots, preprocess, valid_input
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
    ys_rhythm = np.array(ys_rhythm)
    ys_melody = np.array(ys_melody)

    print(f"Xs: {len(Xs)} tracks; {Xs[0].shape} {Xs[1].shape} {Xs[2].shape} {Xs[3].shape} {Xs[4].shape}")
    print(f"Max values: {np.max(Xs[0])} {np.max(Xs[1])} {np.max(Xs[2])} {np.max(Xs[3])} {np.max(Xs[4])}")
    print(f"ys_rhythm: {ys_rhythm.shape}")
    print(f"ys_melody: {ys_melody.shape}")
    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    model_rhythm = build_model(127, 4)  # TODO: Train encoder only once, train two seperate decoders
    model_melody = build_model(25, 48)

    hist_rhythm = model_rhythm.fit(Xs, ys_rhythm, epochs=3, verbose=1, batch_size=32,
                                   validation_split=0.2, shuffle=True, use_multiprocessing=True, workers=6)
    hist_melody = model_melody.fit(Xs, ys_melody, epochs=3, verbose=1, batch_size=32,
                                   validation_split=0.2, shuffle=True, use_multiprocessing=True, workers=6)

    model_rhythm.save("./src/main/python/smt22/model_rhythm.h5")
    model_melody.save("./src/main/python/smt22/model_melody.h5")

    # model_rhythm = tf.keras.models.load_model("./src/main/python/smt22/model_rhythm.h5")
    # model_melody = tf.keras.models.load_model("./src/main/python/smt22/model_melody.h5")

    score_rhythm = model_rhythm.evaluate(Xs, ys_rhythm, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)
    score_melody = model_melody.evaluate(Xs, ys_melody, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)

    print(f"Rhythm model loss: {score_rhythm[0]:.4f}, accuracy: {score_rhythm[1]:.4f}")
    print(f"Melody model loss: {score_melody[0]:.4f}, accuracy: {score_melody[1]:.4f}")

    plots(hist_rhythm, hist_melody)
