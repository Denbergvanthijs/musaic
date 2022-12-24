import matplotlib.pyplot as plt
import numpy as np


def plots(hist_rhythm, hist_melody):
    """Plots the loss and accuracy of both the rhythm and melody models."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(hist_rhythm.history["loss"], label="Rhythm loss")
    axs[0].plot(hist_rhythm.history["val_loss"], label="Rhythm val loss")
    axs[0].plot(hist_melody.history["loss"], label="Melody loss")
    axs[0].plot(hist_melody.history["val_loss"], label="Melody val loss")

    axs[1].plot(hist_rhythm.history["accuracy"], label="Rhythm accuracy")
    axs[1].plot(hist_rhythm.history["val_accuracy"], label="Rhythm val accuracy")
    axs[1].plot(hist_melody.history["accuracy"], label="Melody accuracy")
    axs[1].plot(hist_melody.history["val_accuracy"], label="Melody val accuracy")

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

    lead_rhythm = X[6]
    lead_melody = X[7].reshape(X[7].shape[0], -1)

    X_processed = [context_rhythms, context_melodies, meta, lead_rhythm, lead_melody]

    if y is not None:
        return X_processed, y[0], y[1]

    return X_processed


def valid_input(X, y_rhythm, y_melody):
    context_rhythms, context_melodies, meta, lead_rhythm, lead_melody = X

    if context_rhythms.shape[-1] != 16:
        return False

    if context_melodies.shape[-1] != 192:
        return False

    if meta.shape[-1] != 10:
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
