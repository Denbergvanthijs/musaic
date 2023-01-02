from keras.regularizers import l2
from keras.models import Model
from keras.layers import (Concatenate, Dense, Dropout, Embedding,
                          GlobalAveragePooling1D, Input,
                          RepeatVector, TimeDistributed, LSTM, Reshape)
from keras_nlp.layers import SinePositionEncoding
from v9.Data.DataGeneratorsTransformer import CombinedGenerator
from keras.callbacks import TensorBoard
from tensorflow import keras
from collections import Counter
from smt22.utils import f1_custom, preprocess, valid_input
from tensorflow.keras.models import Model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.history = {'rhythm_decoder_loss': [], 'melody_decoder_loss': [], 'rhythm_decoder_accuracy': [], 'melody_decoder_accuracy': []}

    def on_batch_end(self, batch, logs={}):
        self.history['rhythm_decoder_loss'].append(logs.get('rhythm_decoder_loss'))
        self.history['melody_decoder_loss'].append(logs.get('melody_decoder_loss'))
        self.history['rhythm_decoder_accuracy'].append(logs.get('rhythm_decoder_accuracy'))
        self.history['melody_decoder_accuracy'].append(logs.get('melody_decoder_accuracy'))


def rhythm_encoder(input, name):

    rhythm_attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, name=name)(input, input)
    rhythm_attention = keras.layers.LayerNormalization(epsilon=1e-6)(rhythm_attention)

    rhythm_attention = keras.layers.Dropout(0.2)(rhythm_attention)
    res = rhythm_attention + input

    # Feed Forward Part
    rhythm_attention = Dense(12, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    rhythm_attention = Dropout(0.1)(rhythm_attention)
    rhythm_attention = keras.layers.LayerNormalization(epsilon=1e-6)(rhythm_attention)
    return rhythm_attention + res


def melody_encoder(input, name):

    melody_attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=52, name=name)(input, input)
    melody_attention = keras.layers.LayerNormalization(epsilon=1e-6)(melody_attention)
    melody_attention = keras.layers.Dropout(0.2)(melody_attention)
    res = melody_attention + input

    # Feed Forward Part
    melody_attention = Dense(12, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    melody_attention = Dropout(0.2)(melody_attention)
    melody_attention = keras.layers.LayerNormalization(epsilon=1e-6)(melody_attention)
    return melody_attention + res


def meta_encoder(process_meta: bool = True):
    meta_input = Input(shape=((7 if process_meta else 10),), name="meta_input")
    meta_dense1 = Dense(32, activation="relu", name="meta_dense1", kernel_regularizer=l2(0.001))(meta_input)
    meta_dense2 = Dense(64, activation="relu", name="meta_dense2", kernel_regularizer=l2(0.001))(meta_dense1)

    return meta_input, meta_dense2


def lead_rhythm_encoder(input, name):

    lead_rhythm_attention = keras.layers.MultiHeadAttention(num_heads=2, key_dim=32, name=name)(input, input)
    lead_rhythm_attention = keras.layers.LayerNormalization(epsilon=1e-6)(lead_rhythm_attention)

    lead_rhythm_attention = keras.layers.Dropout(0.1)(lead_rhythm_attention)
    res = lead_rhythm_attention + input

    # Feed Forward Part
    lead_rhythm_attention = Dense(12, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    lead_rhythm_attention = Dropout(0.2)(lead_rhythm_attention)
    lead_rhythm_attention = keras.layers.LayerNormalization(epsilon=1e-6)(lead_rhythm_attention)
    return lead_rhythm_attention + res


def lead_melody_encoder(input, name):

    lead_melody_attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=52, name=name)(input, input)
    lead_melody_attention = keras.layers.LayerNormalization(epsilon=1e-6)(lead_melody_attention)
    lead_melody_attention = keras.layers.Dropout(0.2)(lead_melody_attention)
    res = lead_melody_attention + input

    # Feed Forward Part
    lead_melody_attention = Dense(12, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    lead_melody_attention = Dropout(0.2)(lead_melody_attention)
    lead_melody_attention = keras.layers.LayerNormalization(epsilon=1e-6)(lead_melody_attention)
    return lead_melody_attention + res


def rhythm_decoder(output_length: int, n_repeat: int, input_layer, name):

    decoder = keras.layers.MultiHeadAttention(num_heads=4, key_dim=28, name="rhythm_decoder_attention_" + name)(input_layer, input_layer)
    decoder = keras.layers.LayerNormalization(epsilon=1e-6)(decoder)
    res = decoder + input_layer

    # Feed forward
    decoder = Dense(128, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    decoder = Dropout(0.2)(decoder)
    decoder = keras.layers.LayerNormalization(epsilon=1e-6)(decoder)

    return decoder + res


def melody_decoder(output_length: int, n_repeat: int, input_layer, name):

    decoder = keras.layers.MultiHeadAttention(num_heads=8, key_dim=32, name="melody_decoder_attention" + name)(input_layer, input_layer)
    decoder = keras.layers.LayerNormalization(epsilon=1e-6)(decoder)

    res = decoder + input_layer

    # Feed forward
    decoder = Dense(128, activation="relu", name="rhythm_decoder_dense_" + name, kernel_regularizer=l2(0.001))(res)
    decoder = Dropout(0.2)(decoder)
    decoder = keras.layers.LayerNormalization(epsilon=1e-6)(decoder)

    return decoder + res


def build_model(output_length_rhythm: int, n_repeat_rhythm: int, output_length_melody: int, n_repeat_melody: int,
                which: int, nr_of_layers: int = 1, rhythm_test: bool = True):

    if (which == 1):
        if (rhythm_test):
            model = keras.Sequential()
            model.add(Dense(32, activation="relu", input_shape=(267,)))
            model.add(RepeatVector(n_repeat_rhythm))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(TimeDistributed(Dense(output_length_rhythm, activation="relu", name="check1")))
        else:
            model = keras.Sequential()
            model.add(Dense(32, activation="relu", input_shape=(267,)))
            model.add(RepeatVector(n_repeat_melody))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(TimeDistributed(Dense(output_length_melody, activation="relu", name="check1")))

        return Model(inputs=model.input, outputs=model.output)

    elif (which == 2):
        input_layer = Input(shape=(267,1))
        dense_input = Dense(32, activation="relu")(input_layer)
        dense_input = LSTM(20)(dense_input)


        rhythm_model = RepeatVector(n_repeat_rhythm)(dense_input)
        #rhythm_model = LSTM(20,return_sequences=True)(rhythm_model)
        rhythm_model = Dense(64, activation="relu")(rhythm_model)
        rhythm_model = Dropout(0.2)(rhythm_model)
        rhythm_model = TimeDistributed(Dense(output_length_rhythm, activation="sigmoid"), name="rhythm_decoder")(rhythm_model)

        melody_model = RepeatVector(n_repeat_melody)(dense_input)
       # melody_model = LSTM(20,return_sequences=True)(melody_model)
        melody_model = Dense(128, activation="relu")(melody_model)
        melody_model = Dropout(0.2)(melody_model)
        melody_model = TimeDistributed(Dense(output_length_melody, activation="sigmoid"), name="melody_decoder")(melody_model)

        return Model(inputs=input_layer, outputs=rhythm_model), Model(inputs=input_layer, outputs=melody_model)
    elif(which == 4):
        input_layer = Input(shape=(267,1))

        Lstm_input = LSTM(150)(input_layer)
        
        rhythm_model = RepeatVector(n_repeat_melody)(Lstm_input)
        Lstm_input = LSTM(150,return_sequences=True)(rhythm_model)
        rhythm_model = TimeDistributed(Dense(output_length_rhythm, activation="sigmoid"), name="rhythm_decoder")(Lstm_input)

        melody_model = RepeatVector(n_repeat_melody)(Lstm_input)
        Lstm_input = LSTM(150,return_sequences=True)(melody_model)
        melody_model = TimeDistributed(Dense(output_length_melody, activation="sigmoid"), name="rhythm_decoder")(melody_model)
        return Model(inputs=input_layer, outputs=rhythm_model), Model(inputs=input_layer, outputs=melody_model)
       

    else:
        # rhythm
        rhythm_input = Input(shape=(16, ), name="rhythm_input")
        rhythm_embedding = Embedding(input_dim=129, output_dim=12, name="rhythm_embedding")(rhythm_input)
        positional_encoding = SinePositionEncoding()(rhythm_embedding)
        rhythm_embedding_pos = rhythm_embedding + positional_encoding

        # melody
        melody_input = Input(shape=(192,), name="melody_input")
        melody_embedding = Embedding(input_dim=512, output_dim=12, name="melody_embedding")(melody_input)
        positional_encoding = SinePositionEncoding()(melody_embedding)
        melody_embedding_pos = melody_embedding + positional_encoding

        # lead rhythm
        lead_rhythm_input = Input(shape=(4,), name="lead_rhythm_input")
        lead_rhythm_embedding = Embedding(input_dim=129, output_dim=12, name="lead_rhythm_embedding")(lead_rhythm_input)
        positional_encoding = SinePositionEncoding()(lead_rhythm_embedding)
        lead_rhythm_embedding_pos = lead_rhythm_embedding + positional_encoding

        # lead melody
        lead_melody_input = Input(shape=(48,), name="lead_melody_input")
        lead_melody_embedding = Embedding(input_dim=512, output_dim=12, name="lead_melody_embedding")(lead_melody_input)
        positional_encoding = SinePositionEncoding()(lead_melody_embedding)
        lead_melody_embedding_pos = lead_melody_embedding + positional_encoding

        rhythm_attention = rhythm_encoder(rhythm_embedding_pos, "rhythm_attention")
        melody_attention = melody_encoder(melody_embedding_pos, "melody_attention")
        lead_rhythm_attention = lead_rhythm_encoder(lead_rhythm_embedding_pos, "lead_rhythm_attention")
        lead_melody_attention = lead_melody_encoder(lead_melody_embedding_pos, "lead_melody_attention")

        for i in range(nr_of_layers):
            rhythm_attention = rhythm_encoder(rhythm_attention, "rhythm_attention" + str(i))
            melody_attention = melody_encoder(melody_attention, "melody_attention" + str(i))
            lead_rhythm_attention = lead_rhythm_encoder(lead_rhythm_attention, "lead_rhythm_attention" + str(i))
            lead_melody_attention = lead_melody_encoder(lead_melody_attention, "lead_melody_attention" + str(i))
        meta_input, meta_dense = meta_encoder()
        encoder_inputs = [rhythm_input, melody_input, meta_input, lead_rhythm_input, lead_melody_input]

        # Concat rhythm and melody inputs
        concat_context = Concatenate(axis=1, name="concat_context")([rhythm_attention, melody_attention])
        concat_context = GlobalAveragePooling1D(data_format="channels_first")(concat_context)  # Flatten the output
        concat_context = Dense(128, activation="relu", name="dense_context", kernel_regularizer=l2(0.001))(concat_context)

        # Concat leads
        concat_leads = Concatenate(axis=1, name="concat_leads")([lead_rhythm_attention, lead_melody_attention])
        concat_leads = GlobalAveragePooling1D(data_format="channels_first")(concat_leads)  # Flatten the output
        concat_leads = Dense(32, activation="relu", name="dense_leads", kernel_regularizer=l2())(concat_leads)

        # Concat all inputs
        concat = Concatenate(axis=1, name="concat_all")(
            [concat_context, concat_leads, meta_dense])  # Concatenate the outputs of the encoders
        concat = Dropout(0.2)(concat)

        # Decoder

        r_decoder = Dense(128, activation="relu", name="rhythm_decoder_dense1", kernel_regularizer=l2(0.001))(concat)
        r_decoder = RepeatVector(n_repeat_rhythm)(r_decoder)  # Repeat the output n_repeat times

        m_decoder = Dense(128, activation="relu", name="melody_decoder_dense1", kernel_regularizer=l2(0.001))(concat)
        m_decoder = RepeatVector(n_repeat_melody)(m_decoder)  # Repeat the output n_repeat times

        rhythm_dec = rhythm_decoder(output_length_rhythm, n_repeat_rhythm, r_decoder, "rhythm_dec")
        melody_dec = melody_decoder(output_length_melody, n_repeat_melody, m_decoder, "melo_dec")
        for i in range(nr_of_layers):
            rhythm_dec = rhythm_decoder(output_length_rhythm, n_repeat_rhythm, rhythm_dec, "rhythm_dec" + str(i))
            melody_dec = melody_decoder(output_length_melody, n_repeat_melody, melody_dec, "melo_dec" + str(i))

        rhythm_dec = Dense(128, activation="relu", name="rhythm_decoder_dense2", kernel_regularizer=l2(0.001))(rhythm_dec)
        rhythm_dec = TimeDistributed(Dense(output_length_rhythm, activation="softmax", kernel_regularizer=l2(0.001)),
                                     name="rhythm_decoder")(rhythm_dec)  # Output layer

        melody_dec = Dense(128, activation="relu", name="melody_decoder_dense2", kernel_regularizer=l2(0.001))(melody_dec)
        melody_dec = TimeDistributed(Dense(output_length_melody, activation="softmax", kernel_regularizer=l2(0.001)),
                                     name="melody_decoder")(melody_dec)  # Output layer

        return Model(inputs=encoder_inputs, outputs=rhythm_dec), Model(inputs=encoder_inputs, outputs=melody_dec)


def preprocess(X, y=None, process_meta: bool = True):
    """Preprocesses the data for the model.

    Max values for meta are determined based on the maximum value their knob can be set to.

    Min and max are in the order of sorted(meta_keys):
    ['cDens', 'cDepth', 'expression', 'jump', 'pos', 'rDens', 'span', 'tCent', 'ts part 1', 'ts part 2']
    """
    context_rhythms = np.concatenate([x.reshape(x.shape[0], -1) for x in X[:4]], axis=1)
    context_melodies = X[4].reshape(X[4].shape[0], -1)

    meta = X[5]

    if process_meta:
        # Normalise each value of meta by subtracting the minimum value and dividing by the range
        max_values = np.array([1, 5, 1, 12, 1, 8, 30, 80, 4, 4])
        min_values = np.array([0, 1, 0, 0, 0, 0, 1, 40, 0, 0])
        meta = (meta - min_values) / (max_values - min_values)

        # Only select relevant meta data
        # expression (index 2) and ts (index 8 and 9) are not used
        meta = meta[:, [0, 1, 3, 4, 5, 6, 7]]

    lead_rhythm = X[6].reshape(X[6].shape[0], -1)
    lead_melody = X[7].reshape(X[7].shape[0], -1)

    X_processed = [context_rhythms, context_melodies, meta, lead_rhythm, lead_melody]
    X_processed = [x.astype(np.float32) for x in X_processed]  # Necessary for tf.lite

    if y is not None:
        y[0] = np.transpose(y[0], (0, 2, 1))
        y[1] = np.transpose(y[1], (0, 2, 1))
        return X_processed, y[0], y[1]

    return X_processed


def plots(history):
    """Plots the loss and accuracy of both the rhythm and melody models."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history.history["rhythm_decoder_loss"], label="Rhythm loss")
    axs[0].plot(history.history["melody_decoder_loss"], label="Melody loss")
    # axs[0].plot(history.history["val_rhythm_decoder_loss"], label="Rhythm val loss")
    # axs[0].plot(history.history["val_melody_decoder_loss"], label="Melody val loss")

    axs[1].plot(history.history["rhythm_decoder_accuracy"], label="Rhythm accuracy")
    axs[1].plot(history.history["melody_decoder_accuracy"], label="Melody accuracy")
    # axs[1].plot(history.history["val_rhythm_decoder_accuracy"], label="Rhythm val accuracy")
    # axs[1].plot(history.history["val_melody_decoder_accuracy"], label="Melody val accuracy")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    axs[0].legend()
    axs[1].legend()

    plt.show()


def valid_input(X, y_rhythm, y_melody, process_meta: bool = True) -> bool:
    """Checks if the input is valid."""
    context_rhythms, context_melodies, meta, lead_rhythm, lead_melody = X

    if context_rhythms.shape[-1] != 16:
        return False

    if context_melodies.shape[-1] != 192:
        return False

    if meta.shape[-1] != (7 if process_meta else 10):
        return False

    if lead_rhythm.shape[-1] != 4:
        return False

    if lead_melody.shape[-1] != 48:
        return False

    if y_rhythm.shape[-2] != 127:
        return False

    if y_melody.shape[-2] != 25:
        return False

    return True


if __name__ == "__main__":
    # Inputs
    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"

    # Params
    rhythm_context_size = 4
    melody_context_size = 4

    # Generate data
    combined_generator = CombinedGenerator(fp_music, save_conversion_params=False, to_list=False, meta_prep_f=None)

    # Counter of num_pieces
    num_pieces = combined_generator.get_num_pieces()
    for k, v in Counter(num_pieces).items():
        print(f"{v} songs with {k} instruments")

    # Product of num_pieces
    print(f"Total number of tracks: {sum([k * v for k, v in Counter(num_pieces).items()])}")

    data_iter = combined_generator.generate_data(rhythm_context_size=rhythm_context_size,
                                                 melody_context_size=melody_context_size)

    cnt = 0
    ys_rhythm, ys_melody = [], []
    X1, X2, X3, X4, X5 = [], [], [], [], []
    tempx = []
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
            tempx.append(np.concatenate((context_rhythms[row], context_melodies[row],
                                         meta[row], lead_rhythm[row], lead_melody[row]), axis=0))

            ys_rhythm.append(y_rhythm[row])
            ys_melody.append(y_melody[row])

        # if c == 4:  # Early stop for testing
        #     break



# print(f"Max values: {np.max(Xs[0])} {np.max(Xs[1])} {np.max(Xs[2])} {np.max(Xs[3])} {np.max(Xs[4])}")
# print(f"Min values: {np.min(Xs[0])} {np.min(Xs[1])} {np.min(Xs[2])} {np.min(Xs[3])} {np.min(Xs[4])}")
# print(f"ys_rhythm: {ys[0].shape}; ys_melody: {ys[1].shape}")
# print(f"Max values: {np.max(ys[0])} {np.max(ys[1])}")
# print(f"Min values: {np.min(ys[0])} {np.min(ys[1])}")

    print(f"Skipped {cnt} ({cnt/sum(num_pieces)*100:.0f}%) tracks because of wrong shape")

    kiesmodel =3  # 1,2 of 3
    rhythm_test = True  # true or false

    Xs = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5)]
    ys = [np.array(ys_rhythm), np.array(ys_melody)]

    print(f"Xs: {len(Xs)} tracks; {Xs[0].shape} {Xs[1].shape} {Xs[2].shape} {Xs[3].shape} {Xs[4].shape}")

    if (kiesmodel == 1):  # ez sequential model
        # makkelijke sequential model, als rhythm test false is dan test je de melody
        if (rhythm_test):
            Xs = np.array(tempx)
            ys = np.array(ys_rhythm)
        else:
            Xs = np.array(tempx)
            ys = np.array(ys_rhythm)
    elif (kiesmodel == 2 or kiesmodel == 4):  # ez functional model nu met beide outputs tegelijk
        Xs = np.array(tempx)
        ys = [np.array(ys_rhythm), np.array(ys_melody)]
    else:  # transformer model ig
        Xs = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5)]
        ys = [np.array(ys_rhythm), np.array(ys_melody)]

    # TODO: Train encoder only once, train two seperate decoders
    r_model, m_model = build_model(4, 127, 48, 25, kiesmodel, nr_of_layers=0)
    # ^^
    # 1e 4 variabelen zijn de dimensies, 5e vul je kiesmodel in gaat over welk model je kiest 1=seqential, 2=functional, 3=transformer
    # 6e is het aantal layers dat je het transformerblock meegeeft, hoe meer hoe langer het duurt ofc (default=1). De laatste variabele
    # gaat over kiesmodel=1 als rhythm of melodie wilt trainen(default=True).

    fp_logs = os.path.join("./src/main/python/smt22/logs", datetime.now().strftime("%Y%m%d_%H%M%S"))

    tensorboard_cb = TensorBoard(log_dir=fp_logs, histogram_freq=1)

    opt = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    r_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[f1_custom, precision, recall])# , loss_weights=[4, 48])
    r_model.summary()
    m_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[f1_custom, precision, recall])#, loss_weights=[4, 48])
    m_model.summary()

    # keras.utils.plot_model(model, to_file="./src/main/python/smt22/model_rowan.png", show_shapes=True, dpi=300)
    history = LossHistory()
    hist_model_r = r_model.fit(Xs, np.array(ys_rhythm), epochs=1, batch_size=128, validation_split=0.3,
                               shuffle=True, use_multiprocessing=True, callbacks=[history])
    hist_model_m = m_model.fit(Xs, np.array(ys_melody), epochs=1, batch_size=128, validation_split=0.3,
                               shuffle=True, use_multiprocessing=True, callbacks=[history])
    #plots(history)

    prediction_r = r_model.predict(Xs)
    y_pred_rhythm = np.array(prediction_r)
    y_pred_rhythm = (y_pred_rhythm > 0.5).astype(int).flatten()

    prediction_m = m_model.predict(Xs)
    y_pred_melody = np.array(prediction_m)
    y_pred_melody = (y_pred_melody > 0.5).astype(int).flatten()

    # Calculate F1 score
    print(f"F1 Rhythm: {f1_score(np.array(ys_rhythm).flatten(), y_pred_rhythm, average='macro')}")
    print(f"F1 Melody: {f1_score(np.array(ys_melody).flatten(), y_pred_melody, average='macro')}")

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
    tflite_model_rowan = converter.convert()

    with open("./src/main/python/smt22/model.tflite_rowan", 'wb') as file:
        file.write(tflite_model_rowan)

    score = model.evaluate(Xs, ys, verbose=1, batch_size=32, use_multiprocessing=True, workers=6)

    print(f"Rhythm model loss: {score[1]:.4f}; Melody model loss: {score[2]:.4f}")
    print(f"Rhythm model accuracy: {score[3]:.4f}; Melody model accuracy: {score[4]:.4f}")
