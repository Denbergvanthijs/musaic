from collections import Counter

import numpy as np
from Data.DataGeneratorsTransformer import CombinedGenerator

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

    for c, (X, y) in enumerate(data_iter):
        print(len(X), [x.shape for x in X])  # [*rhythm_x, melody_x, meta, rhythm_lead, melody_lead]
        print(len(y), [y.shape for y in y])  # [rhythm_y, melody_y]

        context_rhythms = [x.reshape(x.shape[0], -1) for x in X[:4]]
        context_rhythms = np.concatenate(context_rhythms, axis=1)
        context_melodies = X[4].reshape(X[4].shape[0], -1)

        meta = X[5]

        lead_rhythm = X[6].reshape(X[6].shape[0], -1)
        lead_melody = X[7].reshape(X[7].shape[0], -1)

        for val in [context_rhythms, context_melodies, meta, lead_rhythm, lead_melody]:
            print(val.shape)

        # Combine all five inputs to one with special tokens
        X_processed = np.concatenate([context_rhythms, context_melodies, meta, lead_rhythm, lead_melody], axis=1)
        print(X_processed.shape)

        print(f"y data: {y[0].shape} {y[1].shape}")

        if c == 1:
            break
