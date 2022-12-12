import random
import time
from copy import deepcopy
from typing import List

import numpy as np
import numpy.random as rand
from core import DEFAULT_AI_PARAMS, DEFAULT_META_DATA, DEFAULT_SECTION_PARAMS
from network import NeuralNet


class Transformer(NeuralNet):
    def __init__(self, resources_path: str = None, init_callbacks: List = None) -> None:
        """Initialise the Transformer network.

        Changes compared to NeuralNet:
        - Uses a Transformer instead of an LSTM
        - Changed try-except to if-else for callbacks
        """
        super(Transformer, self).__init__(resources_path, init_callbacks)

        print("[NeuralNet] Initialising...")  # TODO: Change prints to logging
        self.loaded = False  # Set to true when model is loaded

        time_start = time.time()

        print("[NeuralNet] === Using SMT22 model ===")

        self.combinedNet = lambda _: 1  # Also needs a .predict(x) method
        self.chordNet = lambda _: 1  # Also needs a .predict(x) method

        self.vocabulary = {'rhythm': [0, 1, 2, 3, 4],
                           'melody': [0, 1, 2, 3, 4]}

        self.rhythmDict = {}  # conversionParams['rhythm']
        for k, v in list(self.rhythmDict.items()):  # Reverse dict
            self.rhythmDict[v] = k

        self.chordDict = {}  # chordConversionParams['chords']
        for k, v in list(self.chordDict.items()):
            self.chordDict[v] = k

        # Predict some junk data to fully initilise model...
        self.generateBar(**DEFAULT_SECTION_PARAMS, **DEFAULT_AI_PARAMS)

        print(f"[NeuralNet] Neural network loaded in {int(time.time() - time_start)} seconds")

        self.loaded = True  # Set to true when model is loaded

        if init_callbacks:  # Call any callbacks
            # Check if iterable
            if not hasattr(init_callbacks, "__iter__"):
                init_callbacks = [init_callbacks]

            for f in init_callbacks:
                f()

    def generateBar(self, octave: int = 4, **kwargs) -> list:
        """Generate a bar of music.

        Currently just random notes for testing.
        In the future this will be replaced with a call to the Transformer model.
        """
        notes = []
        for i in range(4):
            note = (random.randint(60, 80), i * 24, (i + 1) * 24)
            notes.append(note)

        return notes

    def embedMetaData(self, meta_data: dict = None) -> np.ndarray:
        """Preprocess meta data into a format that can be used by the model.

        Changes from NeuralNet:
        - Removed the predict function from meta embedding, this enables the processed data to be used by the model.

        """
        if not meta_data:
            meta_data = DEFAULT_META_DATA

        values = []
        for k in sorted(meta_data.keys()):
            if k == "ts":
                values.extend([4, 4])
            else:
                values.append(meta_data[k])

        return np.tile(values, (1, 1))

    def makeNote(self, pc, startTick, endTick, octave: int = 4) -> tuple:
        """Convert a pitch class to a note.

        Changes from NeuralNet:
        - Moved function outside convertContextToNotes
        - Added octave parameter, default 4
        """
        nn = 12 * (octave + 1) + pc - 1
        return int(nn), startTick, endTick

    def predictChord(self, notes, pc, sample_mode, melodyContext, metaData, chord_mode="auto",
                     octave: int = 4, tick: int = 0, endTick: int = 96) -> list:
        """Predict a chord using the chordNet.

        Changes from NeuralNet:
        - embedMetaData is called here to replace duplicate code
        - Moved functions outside convertContextToNotes
        """
        meta_data_processed = self.embedMetaData(metaData)

        model_input = [np.array([[pc]]), np.array([[melodyContext]]), meta_data_processed]
        chord_outputs = self.chordNet.predict(x=model_input)

        if sample_mode in ("dist", "top"):
            chord = rand.choice(len(chord_outputs[0]), p=chord_outputs[0])
        else:  # When sample mode is "best"
            chord = np.argmax(chord_outputs[0], axis=-1)

        intervals = self.chordDict[chord]

        if chord_mode == 1:
            intervals = [rand.choice(intervals)]

        for interval in intervals:
            notes.append(self.makeNote(pc + interval - 12, tick, endTick, octave=octave))

        return notes

    def convertContextToNotes(self, rhythmContext, melodyContext, chordContexts, kwargs, octave=4) -> list:
        """Convert a context to a list of notes.

        Changes from NeuralNet:
        - Provide the chord_mode to predictChord when the chord_mode is 'force'
        - Moved functions outside convertContextToNotes
        - Removed the try except block and replaced with if-else to check if we're at the end of the list
        """
        if "meta_data" not in kwargs or kwargs["meta_data"] is None:
            kwargs["meta_data"] = deepcopy(DEFAULT_META_DATA)

        chord_mode = kwargs.get("chord_mode", 1)
        sample_mode = kwargs.get("sample_mode", "top")

        if chord_mode not in ("force", "auto"):
            chord_mode = int(chord_mode)

        ticks_on = [False] * 96
        for i, beat in enumerate(rhythmContext):
            b = self.rhythmDict[beat]
            for onset in b:
                ticks_on[int((i + onset) * 24)] = True

        ticks_start = [i for i in range(96) if ticks_on[i]]

        notes = []
        for i, tick in enumerate(ticks_start):
            # Removed the try except block and replaced with if-else
            ticks_end = ticks_start[i + 1] if i + 1 < len(ticks_start) else 96

            pc = melodyContext[i // 2]

            if chord_mode == "force":
                tonic = 12 + (pc % 12)

                # Changed chord_mode parameter from none given to force
                notes = self.predictChord(notes, tonic, sample_mode, melodyContext, kwargs["meta_data"],
                                          chord_mode=chord_mode, octave=octave, tick=tick, endTick=ticks_end)

            elif chord_mode in (0, 1, "auto"):
                if pc >= 12:
                    # Draw chord intervals...
                    notes = self.predictChord(notes, pc, sample_mode, melodyContext, kwargs["meta_data"],
                                              chord_mode=chord_mode, octave=octave, tick=tick, endTick=ticks_end)
                else:
                    notes.append(self.makeNote(pc, tick, ticks_end, octave=octave))

            else:
                for chord_pc in chordContexts[i // 2]:
                    notes.append(self.makeNote(chord_pc, tick, ticks_end, octave=octave))

        return notes
