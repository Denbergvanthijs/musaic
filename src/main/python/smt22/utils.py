import matplotlib.pyplot as plt
import numpy as np


def plots(history, metric="accuracy"):
    """Plots the loss and accuracy of both the rhythm and melody models."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history.history["rhythm_decoder_loss"], label="Rhythm loss")
    axs[0].plot(history.history["melody_decoder_loss"], label="Melody loss")
    axs[0].plot(history.history["val_rhythm_decoder_loss"], label="Rhythm val loss")
    axs[0].plot(history.history["val_melody_decoder_loss"], label="Melody val loss")

    axs[1].plot(history.history[f"rhythm_decoder_{metric}"], label=f"Rhythm {metric}")
    axs[1].plot(history.history[f"melody_decoder_{metric}"], label=f"Melody {metric}")
    axs[1].plot(history.history[f"val_rhythm_decoder_{metric}"], label=f"Rhythm val {metric}")
    axs[1].plot(history.history[f"val_melody_decoder_{metric}"], label=f"Melody val {metric}")

    axs[0].set_title("Loss")
    axs[1].set_title(f"{metric.capitalize()}")
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

    lead_rhythm = X[6].reshape(X[6].shape[0], -1)
    lead_melody = X[7].reshape(X[7].shape[0], -1)

    X_processed = [context_rhythms, context_melodies, meta, lead_rhythm, lead_melody]
    X_processed = [x.astype(np.float32) for x in X_processed]  # Necessary for tf.lite

    if y is not None:
        # Permute the dimensions of y to be (batch_size, n_repeats, output_shape)
        y[0] = np.transpose(y[0], (0, 2, 1))
        y[1] = np.transpose(y[1], (0, 2, 1))
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

    if y_rhythm.shape[-2] != 127:
        return False

    if y_melody.shape[-2] != 25:
        return False

    return True
