from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from Data.DataGeneratorsTransformer import CombinedGenerator
from tensorflow.keras.layers import (LSTM, Dense, Embedding, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential


def plots(hist_rhythm, hist_melody):
    """Plots the loss and accuracy of both the rhythm and melody models."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(hist_rhythm.history["loss"], label="Rhythm loss")
    axs[0].plot(hist_rhythm.history["val_loss"], label="Rhythm val loss")
    axs[0].plot(hist_melody.history["loss"], label="Melody loss")
    axs[0].plot(hist_melody.history["val_loss"], label="Melody val loss")

    axs[1].plot(hist_rhythm.history["acc"], label="Rhythm accuracy")
    axs[1].plot(hist_rhythm.history["val_acc"], label="Rhythm val accuracy")
    axs[1].plot(hist_melody.history["acc"], label="Melody accuracy")
    axs[1].plot(hist_melody.history["val_acc"], label="Melody val accuracy")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    axs[0].legend()
    axs[1].legend()

    plt.show()


def preprocess(X, y=None):
    """Preprocesses the data for the model"""
    context_rhythms = np.concatenate([x.reshape(x.shape[0], -1) for x in X[:4]], axis=1)
    context_melodies = X[4].reshape(X[4].shape[0], -1)

    meta = X[5]

    lead_rhythm = X[6].reshape(X[6].shape[0], -1)
    lead_melody = X[7].reshape(X[7].shape[0], -1)

    X_processed = np.concatenate([context_rhythms, context_melodies, meta, lead_rhythm, lead_melody], axis=1)

    if y is not None:
        return X_processed, y[0], y[1]

    return X_processed


def build_model(output_length: int, n_repeat: int):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(270,)))
    model.add(RepeatVector(n_repeat))
    model.add(Dense(64, activation="relu"))
    model.add(TimeDistributed(Dense(output_length, activation="softmax")))  # Softmax each seperate output
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


if __name__ == "__main__":
    # Inputs
    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"
    fp_output = "./src/main/python/smt22/"  # "../../Data/music21

    # Params
    rhythm_context_size = 4
    melody_context_size = 4

    # Generate data
    combined_generator = CombinedGenerator(fp_music, save_conversion_params=fp_output, to_list=False, meta_prep_f=None)

    # Counter of num_pieces
    num_pieces = combined_generator.get_num_pieces()
    for k, v in Counter(num_pieces).items():
        print(f"{v} songs with {k} instruments")

    # Product of num_pieces
    print(f"Total number of tracks: {sum([k * v for k, v in Counter(num_pieces).items()])}")

    data_iter = combined_generator.generate_data(rhythm_context_size=rhythm_context_size,
                                                 melody_context_size=melody_context_size)

    # Build model, input is (batch_size, seq_len) where seq_len=270
    # Output is (batch_size, 4, 127)

    model_rhythm = build_model(127, 4)
    model_melody = build_model(25, 48)

    cnt = 0
    Xs, ys_rhythm, ys_melody = [], [], []
    for c, (X, y) in enumerate(data_iter):
        X, y_rhythm, y_melody = preprocess(X, y)  # Preprocess a single track
        # print(len(X), [x.shape for x in X])  # [*rhythm_x, melody_x, meta, rhythm_lead, melody_lead]
        # print(len(y), [y.shape for y in y])  # [rhythm_y, melody_y]

        if X.shape[1] != 270:
            cnt += 1
            continue

        Xs.append(X)
        ys_rhythm.append(y_rhythm)
        ys_melody.append(y_melody)

    # Concat all tracks, to enable batching with our own batch_size instead of per-track
    # This also improves speed/performance
    Xs = np.concatenate(Xs, axis=0)
    ys_rhythm = np.concatenate(ys_rhythm, axis=0)
    ys_melody = np.concatenate(ys_melody, axis=0)

    hist_rhythm = model_rhythm.fit(Xs, ys_rhythm, epochs=10, verbose=1, batch_size=32,
                                   validation_split=0.2, shuffle=True, use_multiprocessing=True, workers=6)
    hist_melody = model_melody.fit(Xs, ys_melody, epochs=10, verbose=1, batch_size=32,
                                   validation_split=0.2, shuffle=True, use_multiprocessing=True, workers=6)

    model_rhythm.save("./src/main/python/smt22/model_rhythm.h5")
    model_melody.save("./src/main/python/smt22/model_melody.h5")

    # model_rhythm = tf.keras.models.load_model("./src/main/python/smt22/model_rhythm.h5")
    # model_melody = tf.keras.models.load_model("./src/main/python/smt22/model_melody.h5")

    score_rhythm = model_rhythm.evaluate(Xs, ys_rhythm, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)
    score_melody = model_melody.evaluate(Xs, ys_melody, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)

    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")
    print(f"Rhythm model loss: {score_rhythm[0]:.4f}, accuracy: {score_rhythm[1]:.4f}")
    print(f"Melody model loss: {score_melody[0]:.4f}, accuracy: {score_melody[1]:.4f}")

    plots(hist_rhythm, hist_melody)
