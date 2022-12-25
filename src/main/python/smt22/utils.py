import matplotlib.pyplot as plt
import numpy as np


def plots(history):
    """Plots the loss and accuracy of both the rhythm and melody models."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history.history["rhythm_decoder_loss"], label="Rhythm loss")
    axs[0].plot(history.history["melody_decoder_loss"], label="Melody loss")
    axs[0].plot(history.history["val_rhythm_decoder_loss"], label="Rhythm val loss")
    axs[0].plot(history.history["val_melody_decoder_loss"], label="Melody val loss")

    axs[1].plot(history.history["rhythm_decoder_accuracy"], label="Rhythm accuracy")
    axs[1].plot(history.history["melody_decoder_accuracy"], label="Melody accuracy")
    axs[1].plot(history.history["val_rhythm_decoder_accuracy"], label="Rhythm val accuracy")
    axs[1].plot(history.history["val_melody_decoder_accuracy"], label="Melody val accuracy")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    axs[0].legend()
    axs[1].legend()

    plt.show()


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

    lead_rhythm = X[6]
    lead_melody = X[7].reshape(X[7].shape[0], -1)

    X_processed = [context_rhythms, context_melodies, meta, lead_rhythm, lead_melody]

    if y is not None:
        return X_processed, y[0], y[1]

    return X_processed


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

    if y_rhythm.shape[-1] != 127:
        return False

    if y_melody.shape[-1] != 25:
        return False

    return True
