import os
from time import asctime

from Data.DataGeneratorsTransformer import CombinedGenerator
from Nets.MetaEmbedding import MetaEmbedding

if __name__ == "__main__":
    save_dir = asctime().split()
    save_dir = "_".join([*save_dir[0:3], *save_dir[3].split(":")])
    top_dir = "./src/main/python/v9/Trainings/"

    # Inputs
    fp_input = os.path.join(top_dir, "euroAI_lead")  # Path to saved weights and meta
    fp_meta = os.path.join(fp_input, "meta")  # Path to saved meta
    fp_music = "./src/main/python/v9/Data/lessfiles"  # "../../Data/music21"

    # Outputs
    fp_output = os.path.join(top_dir, save_dir)  # Path to save network weights
    fp_logs = os.path.join(fp_output, "logs")  # Path to save logs
    fp_weights = os.path.join(fp_output, "weights")  # Path to save weights

    for fp in [fp_logs, fp_weights]:
        if not os.path.exists(fp):
            os.makedirs(fp)

    # Params
    rhythm_context_size = 4
    melody_context_size = 4

    # Meta
    meta_embedder = MetaEmbedding.from_saved_custom(fp_meta)

    # Change
    combined_generator = CombinedGenerator(fp_music, save_conversion_params=fp_output, to_list=0, meta_prep_f=meta_embedder.predict)
    _ = combined_generator.get_num_pieces()

    data_iter = combined_generator.generate_forever(rhythm_context_size=rhythm_context_size,
                                                    melody_context_size=melody_context_size,
                                                    with_metaData=True)

    # Print output of data generator
    X, y = next(data_iter)

    print(len(X), [x.shape for x in X])
    print(len(y), [y.shape for y in y])
    # x = [*rhythm_x, melody_x, meta, rhythm_lead, melody_lead]
    # y = [rhythm_y, melody_y]
