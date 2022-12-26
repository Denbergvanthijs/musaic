import os
import random
import time
from itertools import product

from app import Engine
from core import DEFAULT_AI_PARAMS, DEFAULT_META_DATA, DEFAULT_SECTION_PARAMS

"""
 == Batch Generator of Songs ==

 Produces N number of V varieties of songs (N*V total), in three folders:

   SAVE_ROOT + "combined/", "lead/", "bass/"

 where each variety is a combination of the PARAMETER_RANGES. Each one of N
 randomly sets the meta_data values.

"""

N = 10
V = 20
ROOT_PATH = "./src/main/python/smt22/generated/"

PARAMETER_RANGES = {"length": [2, 4, 8],  # See core.py for explanation
                    "loop_alt_len": [0, 1],
                    "sample_mode": ["dist"],
                    "chord_mode": [1],
                    "injection_params": [(("qb", "eb"), "maj"),
                                         (("qb",), "maj"),
                                         (("qb", "lb"), "maj"),
                                         (("fb", "eb"), "maj"),
                                         (("tb", "fb"), "maj")]}

META_DATA_RANGES = {"span": (1, 30),  # See core.py for explanation
                    "jump": (0, 12),
                    "cDens": (0, 1),
                    # "cDepth": (1, 5),
                    "tCent": (40, 80),
                    "rDens": (0, 8),
                    "pos": (0, 1)}

if __name__ == "__main__":
    print("STARTING")

    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)

    app = Engine()
    app.start()

    # while not app.networkEngine.isLoaded():
    #     time.sleep(0.01)

    time.sleep(10)

    print(" == Creating instruments...")
    # bass = app.addInstrument(name="bass")
    lead = app.addInstrument(name="chords")

    print(" == Adding sections...")
    # _, bass_sec = bass.newSection(chord_mode=1, octave=2, transpose_octave=-1, length=2, loop_num=8)
    _, lead_sec = lead.newSection()

    print(" == Entering space...")
    print(list(PARAMETER_RANGES.keys()))

    counter = 0
    for loop_num, loop_alt_len, sample_mode, chord_mode, ip in product(*PARAMETER_RANGES.values()):
        params = {**DEFAULT_SECTION_PARAMS, **DEFAULT_AI_PARAMS}
        params["loop_num"] = 8 // loop_num
        params["length"] = loop_num
        params["sample_mode"] = sample_mode
        params["loop_alt_len"] = loop_alt_len
        params["chord_mode"] = chord_mode
        params["octave"] = 4
        # params["lead"] = bass.id_

        lead_sec.changeParameter(**params)

        for _ in range(N):
            # bass_md = {**DEFAULT_META_DATA}
            # for k, (lowerbound, upperbound) in META_DATA_RANGES.items():
            #     bass_md[k] = random.uniform(lowerbound, upperbound)
            # bass_sec.changeParameter(meta_data=bass_md)

            lead_meta_data = {**DEFAULT_META_DATA}
            for k, (lowerbound, upperbound) in META_DATA_RANGES.items():
                lead_meta_data[k] = random.uniform(lowerbound, upperbound)
            lead_sec.changeParameter(meta_data=lead_meta_data)

            # Regenerate...
            # bass.requestGenerateMeasures(gen_all=True)
            lead.requestGenerateMeasures(gen_all=True)
            time.sleep(0.1)

            # while not bass_sec.isGenerated() or not lead_sec.isGenerated():
            while not lead_sec.isGenerated():
                time.sleep(0.1)

            app.exportMidiFile(os.path.abspath(os.path.join(ROOT_PATH, f"melody_{counter:04}.mid")), track_list=None)
            # app.exportMidiFile(os.path.abspath(os.path.join(ROOT_PATH, "bass/", "bass_{counter:04}.mid")), track_list=[bass.id_])
            # app.exportMidiFile(os.path.abspath(os.path.join(ROOT_PATH, "chord/", "chord_{counter:04}.mid")), track_list=[lead.id_])

            print(f"{counter} songs generated")
            counter += 1

    print("DONE")
