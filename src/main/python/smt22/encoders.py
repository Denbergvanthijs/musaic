import tensorflow as tf
from tensorflow.keras.layers import (Dense, Embedding, Input,
                                     MultiHeadAttention, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.models import Model


def rhythm_encoder():
    rhythm_input = Input(shape=(None, ), name="rhythm_input")
    rhythm_embedding = Embedding(input_dim=10_000, output_dim=32, name="rhythm_embedding")(rhythm_input)
    rhythm_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="rhythm_attention")(rhythm_embedding, rhythm_embedding)

    return rhythm_input, rhythm_attention


def melody_encoder():
    melody_input = Input(shape=(None,), name="melody_input")
    melody_embedding = Embedding(input_dim=10_000, output_dim=32, name="melody_embedding")(melody_input)
    melody_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="melody_attention")(melody_embedding, melody_embedding)

    return melody_input, melody_attention


def meta_encoder():
    meta_input = Input(shape=(None,), name="meta_input")
    meta_embedding = Embedding(input_dim=10_000, output_dim=32, name="meta_embedding")(meta_input)
    meta_attention = MultiHeadAttention(num_heads=1, key_dim=32, name="meta_attention")(meta_embedding, meta_embedding)

    return meta_input, meta_attention


def lead_rhythm_encoder():
    lead_rhythm_input = Input(shape=(None,), name="lead_rhythm_input")
    lead_rhythm_embedding = Embedding(input_dim=10_000, output_dim=32, name="lead_rhythm_embedding")(lead_rhythm_input)
    lead_rhythm_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="lead_rhythm_attention")(
        lead_rhythm_embedding, lead_rhythm_embedding)

    return lead_rhythm_input, lead_rhythm_attention


def lead_melody_encoder():
    lead_melody_input = Input(shape=(None,), name="lead_melody_input")
    lead_melody_embedding = Embedding(input_dim=10_000, output_dim=32, name="lead_melody_embedding")(lead_melody_input)
    lead_melody_attention = MultiHeadAttention(num_heads=2, key_dim=32, name="lead_melody_attention")(
        lead_melody_embedding, lead_melody_embedding)

    return lead_melody_input, lead_melody_attention


def build_model(output_length: int, n_repeat: int):
    rhythm_input, rhythm_attention = rhythm_encoder()
    melody_input, melody_attention = melody_encoder()
    meta_input, meta_attention = meta_encoder()
    lead_rhythm_input, lead_rhythm_attention = lead_rhythm_encoder()
    lead_melody_input, lead_melody_attention = lead_melody_encoder()

    # Concatenate the outputs of the models
    encoder_inputs = [rhythm_input, melody_input, meta_input, lead_rhythm_input, lead_melody_input]
    encoder_outputs = [rhythm_attention, melody_attention, meta_attention, lead_rhythm_attention, lead_melody_attention]

    concat = tf.keras.layers.Concatenate(axis=1)(encoder_outputs)  # Concatenate the outputs of the encoders
    concat = tf.keras.layers.Dropout(0.2)(concat)

    # Flatten with global 1D max pooling
    decoder = tf.keras.layers.GlobalAveragePooling1D()(concat)  # Flatten the output
    decoder = tf.keras.layers.Dropout(0.2)(decoder)

    decoder = Dense(256, activation="relu")(decoder)
    decoder = RepeatVector(n_repeat)(decoder)  # Repeat the output n_repeat times
    decoder = Dense(256, activation="relu")(decoder)
    decoder = TimeDistributed(Dense(output_length, activation="softmax"))(decoder)  # Output layer

    # Build the model
    model = Model(inputs=encoder_inputs, outputs=decoder)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
