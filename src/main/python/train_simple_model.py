import os
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from smt22.models import build_simpler_model
from smt22.utils import f1_custom, preprocess, valid_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
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
    Xs = []
    ys_rhythm, ys_melody = [], []
    for c, (X, y) in enumerate(data_iter):
        X, y_rhythm, y_melody = preprocess(X, y, normalisation=False)  # Preprocess a single track

        if not valid_input(X, y_rhythm, y_melody):
            cnt += 1
            continue

        context_rhythms, context_melodies, meta, lead_rhythm, lead_melody = X
        for row in range(context_rhythms.shape[0]):
            Xs.append(np.concatenate((context_rhythms[row], context_melodies[row],
                                      meta[row], lead_rhythm[row], lead_melody[row]), axis=0))

            ys_rhythm.append(y_rhythm[row])
            ys_melody.append(y_melody[row])

        # if c == 4:  # Early stop for testing
        #     break

    Xs = np.array(Xs)

    ys_rhythm = np.array(ys_rhythm)
    ys_melody = np.array(ys_melody)
    ys = [ys_rhythm, ys_melody]

    print(f"Xs: {Xs.shape}")
    print(f"Max values: {np.max(Xs)}, Min values: {np.min(Xs)}")
    print(f"ys_rhythm: {ys[0].shape}; ys_melody: {ys[1].shape}")
    print(f"Max values: {np.max(ys[0])} {np.max(ys[1])}")
    print(f"Min values: {np.min(ys[0])} {np.min(ys[1])}")
    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    # Shape is (batch_size, n_repeats, output_shape)
    fp_logs = os.path.join("./src/main/python/smt22/logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=fp_logs, histogram_freq=1)

    model_rhythm, model_melody = build_simpler_model(
        output_length_rhythm=4, n_repeat_rhythm=127, output_length_melody=48, n_repeat_melody=25)

    fp_logs = os.path.join("./src/main/python/smt22/logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=fp_logs, histogram_freq=1)

    # lr_schedule = PolynomialDecay(initial_learning_rate=0.01, decay_steps=1_000, end_learning_rate=0.0005)
    opt_rhythm = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, clipnorm=None)
    opt_melody = Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, clipnorm=None)

    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    model_rhythm.compile(optimizer=opt_rhythm, loss="categorical_crossentropy", metrics=[f1_custom, precision, recall])
    model_melody.compile(optimizer=opt_melody, loss="categorical_crossentropy", metrics=[f1_custom, precision, recall])

    model_rhythm.summary()
    model_melody.summary()

    # plot_model(model, to_file="./src/main/python/smt22/model_thijs.png", show_shapes=True, dpi=300)

    hist_rhythm = model_rhythm.fit(Xs, np.array(ys_rhythm), epochs=10, verbose=1, batch_size=32, shuffle=True,
                                   use_multiprocessing=True, workers=6, callbacks=[tensorboard_cb])
    hist_melody = model_melody.fit(Xs, np.array(ys_melody), epochs=10, verbose=1, batch_size=32, shuffle=True,
                                   use_multiprocessing=True, workers=6, callbacks=[tensorboard_cb])

    model_rhythm.save("./src/main/python/smt22/model_original_rhythm.h5")
    model_melody.save("./src/main/python/smt22/model_original_melody.h5")

    y_pred_rhythm = model_rhythm.predict(Xs)
    y_pred_rhythm = (y_pred_rhythm > 0.5).astype(int).flatten()

    y_pred_melody = model_melody.predict(Xs)
    y_pred_melody = (y_pred_melody > 0.5).astype(int).flatten()

    # Calculate F1 score
    print(f"F1 Rhythm: {f1_score(ys_rhythm.flatten(), y_pred_rhythm, average='macro')}")
    print(f"F1 Melody: {f1_score(ys_melody.flatten(), y_pred_melody, average='macro')}")

    cm_rhythm = confusion_matrix(np.array(ys_rhythm).flatten(), y_pred_rhythm)
    print(cm_rhythm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rhythm)
    disp.plot()
    plt.show()

    cm_melody = confusion_matrix(np.array(ys_melody).flatten(), y_pred_melody)
    print(cm_melody)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_melody)
    disp.plot()
    plt.show()

    score_rhythm = model_rhythm.evaluate(Xs, np.array(ys_rhythm), verbose=1, batch_size=32, use_multiprocessing=True, workers=6)
    score_melody = model_melody.evaluate(Xs, np.array(ys_melody), verbose=1, batch_size=32, use_multiprocessing=True, workers=6)

    print(f"Rhythm model loss: {score_rhythm[0]:.4f}; Melody model loss: {score_melody[0]:.4f}")
    print(f"Rhythm model accuracy: {score_rhythm[1]:.4f}; Melody model accuracy: {score_melody[1]:.4f}")

    # plots(hist_model, metric="categorical_accuracy")
