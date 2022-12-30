import os
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
from smt22.models import build_model, build_original_rhythm, build_simple_model
from smt22.utils import plots, preprocess, valid_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
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
    X1_1, X1_2, X1_3, X1_4 = [], [], [], []
    X2, X3, X4, X5 = [], [], [], []
    for c, (X, y) in enumerate(data_iter):
        X, y_rhythm, y_melody = preprocess(X, y)  # Preprocess a single track

        if not valid_input(X, y_rhythm, y_melody):
            cnt += 1
            continue

        context_rhythms, context_melodies, meta, lead_rhythm, lead_melody = X
        for row in range(context_rhythms.shape[0]):
            X1_1.append(context_rhythms[row][:4])
            X1_2.append(context_rhythms[row][4:8])
            X1_3.append(context_rhythms[row][8:12])
            X1_4.append(context_rhythms[row][12:])

            X2.append(context_melodies[row])
            X3.append(meta[row])
            X4.append(lead_rhythm[row])
            X5.append(lead_melody[row])

            ys_rhythm.append(y_rhythm[row])
            ys_melody.append(y_melody[row])

        # if c == 4:  # Early stop for testing
        #     break

    Xs = [np.array(X1_1), np.array(X1_2), np.array(X1_3), np.array(X1_4), np.array(X3), np.array(X4)]
    ys = [np.array(ys_rhythm), np.array(ys_melody)]

    print(f"Xs: {len(Xs)} tracks; {Xs[0].shape} {Xs[1].shape} {Xs[2].shape} {Xs[3].shape} {Xs[4].shape} {Xs[5].shape}")
    print(f"Max values: {np.max(Xs[0])} {np.max(Xs[1])} {np.max(Xs[2])} {np.max(Xs[3])} {np.max(Xs[4])} {np.max(Xs[5])}")
    print(f"Min values: {np.min(Xs[0])} {np.min(Xs[1])} {np.min(Xs[2])} {np.min(Xs[3])} {np.min(Xs[4])} {np.min(Xs[5])}")
    print(f"ys_rhythm: {ys[0].shape}; ys_melody: {ys[1].shape}")
    print(f"Max values: {np.max(ys[0])} {np.max(ys[1])}")
    print(f"Min values: {np.min(ys[0])} {np.min(ys[1])}")
    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    # Shape is (batch_size, n_repeats, output_shape)
    # model = build_simple_model(output_length_rhythm=4, n_repeat_rhythm=127, output_length_melody=48, n_repeat_melody=25)
    model = build_original_rhythm(output_length_rhythm=4, n_repeat_rhythm=127)

    fp_logs = os.path.join("./src/main/python/smt22/logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=fp_logs, histogram_freq=1)

    # lr_schedule = PolynomialDecay(initial_learning_rate=0.01, decay_steps=1_000, end_learning_rate=0.0005)
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=None)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    model.summary()
    plot_model(model, to_file="./src/main/python/smt22/model_thijs.png", show_shapes=True, dpi=300)

    hist_model = model.fit(Xs, np.array(ys_rhythm), epochs=10, verbose=1, batch_size=24, validation_split=0.15,
                           shuffle=True, use_multiprocessing=True, workers=6, callbacks=[tensorboard_cb])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open("./src/main/python/smt22/model.tflite", 'wb') as file:
        file.write(tflite_model)

    score = model.evaluate(Xs, np.array(ys_rhythm), verbose=1, batch_size=32, use_multiprocessing=True, workers=6)
    print(f"Rhythm model loss: {score[1]:.4f}; Melody model loss: {score[2]:.4f}")
    print(f"Rhythm model accuracy: {score[3]:.4f}; Melody model accuracy: {score[4]:.4f}")

    plots(hist_model)
