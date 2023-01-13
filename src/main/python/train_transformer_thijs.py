import os
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from smt22.models import build_model, build_simple_model
from smt22.utils import f1_custom, plots, preprocess, valid_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.utils import plot_model
from v9.Data.DataGeneratorsTransformer import CombinedGenerator

if __name__ == "__main__":
    USE_SIMPLE_MODEL = False
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
    Xs = []
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

            # Flatten all in X0
            if USE_SIMPLE_MODEL:
                Xs.append(np.concatenate((context_rhythms[row], context_melodies[row],
                          meta[row], lead_rhythm[row], lead_melody[row]), axis=0))

            ys_rhythm.append(y_rhythm[row])
            ys_melody.append(y_melody[row])

        # if c == 4:  # Early stop for testing
        #     break

    if USE_SIMPLE_MODEL:
        Xs = np.array(Xs)
    else:
        Xs = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5)]

    ys_rhythm = np.array(ys_rhythm)
    ys_melody = np.array(ys_melody)
    ys = [ys_rhythm, ys_melody]

    print(f"Xs: {len(Xs)} tracks; {Xs[0].shape} {Xs[1].shape} {Xs[2].shape} {Xs[3].shape} {Xs[4].shape}")
    print(f"Max values: {np.max(Xs[0])} {np.max(Xs[1])} {np.max(Xs[2])} {np.max(Xs[3])} {np.max(Xs[4])}")
    print(f"Min values: {np.min(Xs[0])} {np.min(Xs[1])} {np.min(Xs[2])} {np.min(Xs[3])} {np.min(Xs[4])}")
    print(f"ys_rhythm: {ys[0].shape}; ys_melody: {ys[1].shape}")
    print(f"Max values: {np.max(ys[0])} {np.max(ys[1])}")
    print(f"Min values: {np.min(ys[0])} {np.min(ys[1])}")
    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    # Shape is (batch_size, n_repeats, output_shape)
    if USE_SIMPLE_MODEL:
        model = build_simple_model(n_classes_rhythm=127, n_notes_rhythm=4, n_classes_melody=25, n_notes_melody=48)
    else:
        model = build_model(n_classes_rhythm=127, n_notes_rhythm=4, n_classes_melody=25, n_notes_melody=48)

    fp_logs = os.path.join("./src/main/python/smt22/logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=fp_logs, histogram_freq=1)

    # lr_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=1_000, end_learning_rate=0.0005)
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=None)
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[f1_custom, precision, recall], loss_weights=[0.01, 0.99])

    model.summary()
    plot_model(model, to_file="./src/main/python/smt22/model_thijs.png", show_shapes=True, dpi=300)

    hist_model = model.fit(Xs, ys, epochs=10, verbose=1, batch_size=16, validation_split=0.10,
                           shuffle=True, use_multiprocessing=True, callbacks=[tensorboard_cb])

    # Plot confusion matrix
    y_pred = model.predict(Xs)
    y_pred_rhythm = (y_pred[0] > 0.5).astype(int).flatten()
    y_pred_melody = (y_pred[1] > 0.5).astype(int).flatten()

    # Calculate F1 scores
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

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open("./src/main/python/smt22/model.tflite", 'wb') as file:
        file.write(tflite_model)

    score = model.evaluate(Xs, ys, verbose=1, batch_size=32, use_multiprocessing=True)
    print(f"Rhythm model loss: {score[1]:.4f}; Melody model loss: {score[2]:.4f}")
    print(f"Rhythm model accuracy: {score[3]:.4f}; Melody model accuracy: {score[4]:.4f}")

    plots(hist_model, "f1_custom")
