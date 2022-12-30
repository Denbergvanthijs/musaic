from keras_nlp.layers import SinePositionEncoding
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Concatenate, Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, Input,
                                     MultiHeadAttention, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def rhythm_encoder():
    rhythm_input = Input(shape=(16, ), name="rhythm_input")
    rhythm_embedding = Embedding(input_dim=129, output_dim=300, name="rhythm_embedding")(rhythm_input)

    positional_encoding = SinePositionEncoding()(rhythm_embedding)
    rhythm_embedding_pos = rhythm_embedding + positional_encoding

    rhythm_attention = MultiHeadAttention(num_heads=4, key_dim=32, name="rhythm_attention")(rhythm_embedding_pos, rhythm_embedding_pos)
    rhythm_attention = BatchNormalization()(rhythm_attention)

    return rhythm_input, rhythm_attention


def melody_encoder():
    melody_input = Input(shape=(192,), name="melody_input")
    melody_embedding = Embedding(input_dim=512, output_dim=300, name="melody_embedding")(melody_input)

    positional_encoding = SinePositionEncoding()(melody_embedding)
    melody_embedding_pos = melody_embedding + positional_encoding

    melody_attention = MultiHeadAttention(num_heads=8, key_dim=52, name="melody_attention")(melody_embedding_pos, melody_embedding_pos)
    melody_attention = BatchNormalization()(melody_attention)

    return melody_input, melody_attention


def meta_encoder(process_meta: bool = True):
    meta_input = Input(shape=((7 if process_meta else 10),), name="meta_input")
    meta_dense1 = Dense(32, activation="relu", name="meta_dense1", kernel_regularizer=l2(0.00001))(meta_input)
    meta_dense2 = Dense(64, activation="relu", name="meta_dense2", kernel_regularizer=l2(0.00001))(meta_dense1)

    return meta_input, meta_dense2


def lead_rhythm_encoder():
    lead_rhythm_input = Input(shape=(4,), name="lead_rhythm_input")
    lead_rhythm_embedding = Embedding(input_dim=129, output_dim=150, name="lead_rhythm_embedding")(lead_rhythm_input)

    positional_encoding = SinePositionEncoding()(lead_rhythm_embedding)
    lead_rhythm_embedding_pos = lead_rhythm_embedding + positional_encoding

    lead_rhythm_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="lead_rhythm_attention")(
        lead_rhythm_embedding_pos, lead_rhythm_embedding_pos)
    lead_rhythm_attention = BatchNormalization()(lead_rhythm_attention)

    return lead_rhythm_input, lead_rhythm_attention


def lead_melody_encoder():
    lead_melody_input = Input(shape=(48,), name="lead_melody_input")
    lead_melody_embedding = Embedding(input_dim=512, output_dim=150, name="lead_melody_embedding")(lead_melody_input)

    positional_encoding = SinePositionEncoding()(lead_melody_embedding)
    lead_melody_embedding_pos = lead_melody_embedding + positional_encoding

    lead_melody_attention = MultiHeadAttention(num_heads=4, key_dim=52, name="lead_melody_attention")(
        lead_melody_embedding_pos, lead_melody_embedding_pos)
    lead_melody_attention = BatchNormalization()(lead_melody_attention)

    return lead_melody_input, lead_melody_attention


def rhythm_decoder(output_length: int, n_repeat: int, input_layer):
    decoder = Dense(128, activation="relu", name="rhythm_decoder_dense1", kernel_regularizer=l2(0.00001))(input_layer)
    decoder = RepeatVector(n_repeat)(decoder)  # Repeat the output n_repeat times

    decoder = MultiHeadAttention(num_heads=4, key_dim=28, name="rhythm_decoder_attention")(decoder, decoder)
    decoder = BatchNormalization()(decoder)

    decoder = Dense(128, activation="relu", name="rhythm_decoder_dense2", kernel_regularizer=l2(0.00001))(decoder)
    decoder = TimeDistributed(Dense(output_length, activation="softmax", kernel_regularizer=l2(0.00001)),
                              name="rhythm_decoder")(decoder)  # Output layer

    return decoder


def melody_decoder(output_length: int, n_repeat: int, input_layer):
    decoder = Dense(128, activation="relu", name="melody_decoder_dense1", kernel_regularizer=l2(0.00001))(input_layer)
    decoder = RepeatVector(n_repeat)(decoder)  # Repeat the output n_repeat times

    decoder = MultiHeadAttention(num_heads=8, key_dim=32, name="melody_decoder_attention")(decoder, decoder)
    decoder = BatchNormalization()(decoder)

    decoder = Dense(128, activation="relu", name="melody_decoder_dense2", kernel_regularizer=l2(0.00001))(decoder)
    decoder = TimeDistributed(Dense(output_length, activation="softmax", kernel_regularizer=l2(0.00001)),
                              name="melody_decoder")(decoder)  # Output layer

    return decoder


def build_model(output_length_rhythm: int, n_repeat_rhythm: int, output_length_melody: int, n_repeat_melody: int):
    rhythm_input, rhythm_attention = rhythm_encoder()
    melody_input, melody_attention = melody_encoder()
    meta_input, meta_dense = meta_encoder()
    lead_rhythm_input, lead_rhythm_attention = lead_rhythm_encoder()
    lead_melody_input, lead_melody_attention = lead_melody_encoder()

    encoder_inputs = [rhythm_input, melody_input, meta_input, lead_rhythm_input, lead_melody_input]

    # Concat rhythm and melody inputs
    concat_context = Concatenate(axis=1, name="concat_context")([rhythm_attention, melody_attention])
    concat_context = GlobalAveragePooling1D(data_format="channels_last")(concat_context)  # Flatten the output
    concat_context = Dense(128, activation="relu", name="dense_context", kernel_regularizer=l2(0.00001))(concat_context)

    # Concat leads
    concat_leads = Concatenate(axis=1, name="concat_leads")([lead_rhythm_attention, lead_melody_attention])
    concat_leads = GlobalAveragePooling1D(data_format="channels_last")(concat_leads)  # Flatten the output
    concat_leads = Dense(32, activation="relu", name="dense_leads", kernel_regularizer=l2(0.00001))(concat_leads)

    # Concat all inputs
    concat = Concatenate(axis=1, name="concat_all")([concat_context, concat_leads, meta_dense])  # Concatenate the outputs of the encoders
    concat = Dropout(0.1)(concat)

    # Decoder
    rhythm_dec = rhythm_decoder(output_length_rhythm, n_repeat_rhythm, concat)
    melody_dec = melody_decoder(output_length_melody, n_repeat_melody, concat)

    return Model(inputs=encoder_inputs, outputs=[rhythm_dec, melody_dec])


def build_simple_model(output_length_rhythm: int, n_repeat_rhythm: int, output_length_melody: int, n_repeat_melody: int):
    input_layer = Input(shape=(267,))
    dense_input = Dense(32, activation="relu")(input_layer)

    rhythm_model = RepeatVector(n_repeat_rhythm)(dense_input)
    rhythm_model = Dense(64, activation="relu")(rhythm_model)
    rhythm_model = TimeDistributed(Dense(output_length_rhythm, activation="softmax"), name="rhythm_decoder")(rhythm_model)

    melody_model = RepeatVector(n_repeat_melody)(dense_input)
    melody_model = Dense(64, activation="relu")(melody_model)
    melody_model = TimeDistributed(Dense(output_length_melody, activation="softmax"), name="melody_decoder")(melody_model)

    return Model(inputs=input_layer, outputs=[rhythm_model, melody_model])


def rhythm_encoder_original():
    input_layer = Input(shape=(None, ),)
    embed_layer = Embedding(input_dim=129, output_dim=12)(input_layer)
    lstm_layer = Bidirectional(LSTM(12), merge_mode="concat")(embed_layer)
    out_layer = Dense(8)(lstm_layer)

    return input_layer, out_layer


def meta_encoder_original():
    input_layer = Input(shape=(7,))
    dense_1 = Dense(9, activation="relu")(input_layer)
    out_layer = Dense(9, activation="softmax")(dense_1)

    return input_layer, out_layer


def build_original_rhythm(output_length_rhythm: int, n_repeat_rhythm: int):
    rhythm_input_1, rhythm_out_1 = rhythm_encoder_original()
    rhythm_input_2, rhythm_out_2 = rhythm_encoder_original()
    rhythm_input_3, rhythm_out_3 = rhythm_encoder_original()
    rhythm_input_4, rhythm_out_4 = rhythm_encoder_original()

    lead_rhythm_input, lead_rhythm_out = rhythm_encoder_original()

    meta_input, meta_out = meta_encoder_original()

    embeddings_processed = Concatenate()([rhythm_out_1, rhythm_out_2, rhythm_out_3, rhythm_out_4, meta_out, lead_rhythm_out])
    repeat_layer = RepeatVector(n_repeat_rhythm)(embeddings_processed)

    decoded = LSTM(10, return_sequences=True, name="rhythm_lstm")(repeat_layer)
    preds = TimeDistributed(Dense(output_length_rhythm, activation="softmax"), name="rhythm_decoder")(decoded)

    inputs = [rhythm_input_1, rhythm_input_2, rhythm_input_3, rhythm_input_4, meta_input, lead_rhythm_input]
    return Model(inputs=inputs, outputs=preds)
