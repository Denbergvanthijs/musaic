from keras_nlp.layers import SinePositionEncoding
from tensorflow.keras.layers import (Concatenate, Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, Input,
                                     LayerNormalization, MultiHeadAttention,
                                     RepeatVector, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def rhythm_encoder():
    rhythm_input = Input(shape=(16, ), name="rhythm_input")
    rhythm_embedding = Embedding(input_dim=256, output_dim=96, name="rhythm_embedding")(rhythm_input)

    positional_encoding = SinePositionEncoding()(rhythm_embedding)
    rhythm_embedding_pos = rhythm_embedding + positional_encoding

    rhythm_attention = MultiHeadAttention(num_heads=4, key_dim=32, name="rhythm_attention")(rhythm_embedding_pos, rhythm_embedding_pos)
    rhythm_attention = LayerNormalization(epsilon=1e-6)(rhythm_attention)

    return rhythm_input, rhythm_attention


def melody_encoder():
    melody_input = Input(shape=(192,), name="melody_input")
    melody_embedding = Embedding(input_dim=256, output_dim=96, name="melody_embedding")(melody_input)

    positional_encoding = SinePositionEncoding()(melody_embedding)
    melody_embedding_pos = melody_embedding + positional_encoding

    melody_attention = MultiHeadAttention(num_heads=8, key_dim=32, name="melody_attention")(melody_embedding_pos, melody_embedding_pos)
    melody_attention = LayerNormalization(epsilon=1e-6)(melody_attention)

    return melody_input, melody_attention


def meta_encoder():
    meta_input = Input(shape=(10,), name="meta_input")
    meta_dense1 = Dense(64, activation="relu", name="meta_dense1")(meta_input)
    meta_dense2 = Dense(32, activation="relu", name="meta_dense2")(meta_dense1)

    return meta_input, meta_dense2


def lead_rhythm_encoder():
    lead_rhythm_input = Input(shape=(4,), name="lead_rhythm_input")
    lead_rhythm_embedding = Embedding(input_dim=256, output_dim=64, name="lead_rhythm_embedding")(lead_rhythm_input)

    positional_encoding = SinePositionEncoding()(lead_rhythm_embedding)
    lead_rhythm_embedding_pos = lead_rhythm_embedding + positional_encoding

    lead_rhythm_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="lead_rhythm_attention")(
        lead_rhythm_embedding_pos, lead_rhythm_embedding_pos)
    lead_rhythm_attention = LayerNormalization(epsilon=1e-6)(lead_rhythm_attention)

    return lead_rhythm_input, lead_rhythm_attention


def lead_melody_encoder():
    lead_melody_input = Input(shape=(48,), name="lead_melody_input")
    lead_melody_embedding = Embedding(input_dim=256, output_dim=64, name="lead_melody_embedding")(lead_melody_input)

    positional_encoding = SinePositionEncoding()(lead_melody_embedding)
    lead_melody_embedding_pos = lead_melody_embedding + positional_encoding

    lead_melody_attention = MultiHeadAttention(num_heads=4, key_dim=32, name="lead_melody_attention")(
        lead_melody_embedding_pos, lead_melody_embedding_pos)
    lead_melody_attention = LayerNormalization(epsilon=1e-6)(lead_melody_attention)

    return lead_melody_input, lead_melody_attention


def build_model(output_length: int, n_repeat: int):
    rhythm_input, rhythm_attention = rhythm_encoder()
    melody_input, melody_attention = melody_encoder()
    meta_input, meta_dense = meta_encoder()
    lead_rhythm_input, lead_rhythm_attention = lead_rhythm_encoder()
    lead_melody_input, lead_melody_attention = lead_melody_encoder()

    encoder_inputs = [rhythm_input, melody_input, meta_input, lead_rhythm_input, lead_melody_input]

    # Concat rhythm and melody inputs
    concat_context = Concatenate(axis=1, name="concat_context")([rhythm_attention, melody_attention])
    concat_context = GlobalAveragePooling1D()(concat_context)  # Flatten the output
    concat_context = Dense(256, activation="relu", name="dense_context")(concat_context)

    # Concat leads
    concat_leads = Concatenate(axis=1, name="concat_leads")([lead_rhythm_attention, lead_melody_attention])
    concat_leads = GlobalAveragePooling1D()(concat_leads)  # Flatten the output
    concat_leads = Dense(256, activation="relu", name="dense_leads")(concat_leads)

    # Concat all inputs
    concat = Concatenate(axis=1, name="concat_all")([concat_context, concat_leads, meta_dense])  # Concatenate the outputs of the encoders
    concat = Dropout(0.2)(concat)

    # Decoder
    decoder = Dense(128, activation="relu", name="decoder_dense1")(concat)
    decoder = Dropout(0.2)(decoder)
    decoder = RepeatVector(n_repeat)(decoder)  # Repeat the output n_repeat times

    decoder = MultiHeadAttention(num_heads=4, key_dim=32, name="decoder_attention")(decoder, decoder)
    decoder = LayerNormalization(epsilon=1e-6)(decoder)

    decoder = Dense(128, activation="relu", name="decoder_dense2")(decoder)
    decoder = TimeDistributed(Dense(output_length, activation="softmax", name="decoder_dense3"))(decoder)  # Output layer

    # Build the model
    model = Model(inputs=encoder_inputs, outputs=decoder)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
