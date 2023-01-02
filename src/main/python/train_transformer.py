from collections import Counter

from v9.Data.DataGeneratorsTransformer import CombinedGenerator

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

    # Print the first iteration of the data
    X, y = next(data_iter)

    print(len(X), [x.shape for x in X])  # [*rhythm_x, melody_x, meta, rhythm_lead, melody_lead]
    print(len(y), [y.shape for y in y])  # [rhythm_y, melody_y]
    print(y[0].min(), y[0].max())
    print(y[1].min(), y[1].max())
    print(y[0].argmax(axis=2)[0])
    print(y[1].argmax(axis=2)[0])
