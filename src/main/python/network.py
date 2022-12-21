import multiprocessing
import os
import pickle
import random
import time
from copy import deepcopy
from typing import Union

import numpy as np
import numpy.random as rand
from core import DEFAULT_AI_PARAMS, DEFAULT_META_DATA, DEFAULT_SECTION_PARAMS

RANDOM = 0
VER_9 = 1
EUROAI = 2
SMT22 = 3

PLAYER = 2

if PLAYER != RANDOM:
    from v9.Nets.ChordNetwork import ChordNetwork
    from v9.Nets.CombinedNetworkEuro import CombinedNetwork
    from v9.Nets.MetaEmbeddingEuro import MetaEmbedding
    from v9.Nets.MetaPredictorEuro import MetaPredictor


class RandomPlayer():
    """For testing purpose only!"""

    def __init__(self):
        print("[RandomPlayer] === Using RANDOM PLAYER for testing ===")

    def generateBar(self, **kwargs: dict) -> list:
        """Generates a random bar of music. Returns a list of MIDI notes in the form (pitch, start tick, end tick)."""
        notes = []
        for i in range(4):
            note = (random.randint(60, 80), i * 24, (i + 1) * 24)  # (pitch, start tick, end tick)
            notes.append(note)

        return notes


class NeuralNet():
    def __init__(self, resources_path=None, init_callbacks=None) -> None:
        """Initialises either the V9 or EuroAI neural network.

        Changes compared to original implementation:
        - Removed resources_path parameter, as it is not used
        - Changed try-except to if-else for callbacks
        """
        print("[NeuralNet]", "Initialising...")

        self.loaded = False
        time_start = time.time()

        if PLAYER == VER_9:
            fp_training_data = "./src/main/resources/base/v9_lead/"
        elif PLAYER == EUROAI:
            fp_training_data = "./src/main/resources/base/euroAI/"
        else:
            raise (f"[NeuralNet] Unknown player initialised ({PLAYER}). Aborting")

        print(f"[NeuralNet] === Using {'VER9' if PLAYER == VER_9 else 'EUROAI'} ===")

        with open(os.path.join(fp_training_data, "DataGenerator.conversion_params"), "rb") as f:
            params_conversion = pickle.load(f)

        with open(os.path.join(fp_training_data, "ChordGenerator.conversion_params"), "rb") as f:
            params_conversion_chord = pickle.load(f)

        self.rhythmDict = params_conversion["rhythm"]
        for k, v in list(self.rhythmDict.items()):  # Reverse dict
            self.rhythmDict[v] = k

        self.chordDict = params_conversion_chord["chords"]
        for k, v in list(self.chordDict.items()):  # Reverse dict
            self.chordDict[v] = k

        fp_chords = os.path.join(fp_training_data, "chord")
        self.model_chords = ChordNetwork.from_saved_custom(fp_chords, load_melody_encoder=True)

        fp_meta = os.path.join(fp_training_data, "meta")
        self.meta_embedder = MetaEmbedding.from_saved_custom(fp_meta)  # Used to preprocess meta data to input into model
        meta_predictor = MetaPredictor.from_saved_custom(fp_meta)

        fp_weights = os.path.join(fp_training_data, "weights")
        self.model = CombinedNetwork.from_saved_custom(fp_weights, meta_predictor, generation=True, compile_now=False)

        self.vocabulary = {"rhythm": self.model.params["rhythm_net_params"][2],
                           "melody": self.model.params["melody_net_params"][3]}

        self.note_durations_dict = {"qb": [self.rhythmDict[(0.0,)]] * 2,  # Quarter
                                    "lb": [self.rhythmDict[()]],  # Long
                                    "eb": [self.rhythmDict[(0.0, 0.5)], self.rhythmDict[(0.5,)]] * 2,  # Eighth
                                    "fb": [self.rhythmDict[(0.0, 0.25, 0.5, 0.75)],  # Sixteenth
                                           self.rhythmDict[(0.0, 0.25, 0.5)],
                                           self.rhythmDict[(0.5, 0.75)]] * 2,
                                    "tb": [self.rhythmDict[(0.0, 0.333, 0.6667)],  # Triplet
                                           self.rhythmDict[(0.3333, 0.6667)]] * 2, }

        self.scales_dict = {"maj": [1, 3, 5, 6, 8, 10, 12],  # Major
                            "min": [1, 3, 4, 6, 8, 10, 11],  # Minor
                            "pen": [1, 4, 6, 8, 11],  # Pentatonic
                            "5th": [1, 8]}  # 5th

        # Predict some junk data to fully initialise model...
        _ = self.generateBar(**DEFAULT_SECTION_PARAMS, **DEFAULT_AI_PARAMS)  # TODO: investigate if this is necessary

        print(f"[NeuralNet] Neural network loaded in {int(time.time() - time_start)} seconds")

        self.loaded = True  # Set to true when model is loaded

        if init_callbacks:  # Call any callbacks
            if not hasattr(init_callbacks, "__iter__"):  # Check if iterable
                init_callbacks = [init_callbacks]

            for f in init_callbacks:
                f()

    def generateBar(self, octave: int = 4, **kwargs) -> list:
        """Generates a bar of music.

        Expecting...
            - "lead_bar"
            - "prev_bars"
            - "sample_mode"
            - "chord_mode"
            - "lead_mode"
            - "context_mode"
            - "injection_params"
            - "meta_data"
            - "octave"
        """
        # Preprocess input
        context_rhythms, context_melodies = self.getContexts(kwargs)
        meta_data_embedded = self.embedMetaData(kwargs["meta_data"])
        lead_rhythm, lead_melody = self.getLead(kwargs, context_rhythms, context_melodies)

        # TEMP: Write model input to file for debugging
        with open("model_input.txt", "w+") as f:
            for name, i in zip(["context_rhythms", "context_melodies", "meta_data_embedded", "lead_rhythm", "lead_melody"],
                               [context_rhythms, context_melodies, meta_data_embedded, lead_rhythm, lead_melody]):
                f.write(f"{name}:\n{i}\n\n")

        model_input = [*context_rhythms, context_melodies, meta_data_embedded, lead_rhythm, lead_melody]
        model_output = self.model.predict(x=model_input)

        # Postprocess output
        sampled_rhythm, sampled_melody, sampled_chords = self.sampleOutput(model_output, kwargs)

        return self.convertContextToNotes(sampled_rhythm[0], sampled_melody[0], sampled_chords, kwargs, octave=octave)

    def preprocessMetaData(self, meta_data: dict = None) -> np.ndarray:
        """Preprocess meta data into a format that can be used by the model.

        Changes compared to original implementation:
        - Seperated the preprocessing from the embedding, resulting in this new function
        """
        if not meta_data:
            meta_data = DEFAULT_META_DATA

        values = []
        for k in sorted(meta_data.keys()):
            if k == "ts":
                values.extend([4, 4])  # TODO: make function more general with Fraction(metaData[k], _normalize=False)
            else:
                values.append(meta_data[k])

        return np.tile(values, (1, 1))

    def embedMetaData(self, meta_data: dict = None) -> np.ndarray:
        """Embeds meta data into a format that can be used by the model.

        Changes compared to original implementation:
        - Seperated the preprocessing from the embedding, resulting in new function preprocessMetaData
        """
        return self.meta_embedder.predict(self.preprocessMetaData(meta_data))

    def getContexts(self, kwargs: dict) -> list:
        """Returns rhythm and melody contexts.

        Either from previous bars (real) or from injection params (inject).
        Currently set to the four most recent bars for real mode.
        Or four injected bars for inject mode.

        The user can set the 'context_mode' and 'injection_params' in the MusAIc GUI.

        Changes compared to original implementation:
        - Moved getters for kwargs within if-else
        - Seperated dict creation from dict lookup, moved dict creation to __init__ so it only happens once
        """
        context_mode = kwargs.get("context_mode", None)  # "inject" new measures, "real" use previous measures

        if context_mode == "inject":  # Inject new measures
            # *2 gives extra weight to non-empty beats

            note_durations, scale = kwargs.get("injection_params", DEFAULT_AI_PARAMS["injection_params"])

            rhythm_pool = []
            for note_duration in note_durations:
                rhythm_pool.extend(self.note_durations_dict[note_duration])  # Add selected note durations to pool

            context_rhythms = [np.random.choice(rhythm_pool, size=(1, 4)) for _ in range(4)]

            melody_pool = self.scales_dict[scale]  # Add selected scale to pool
            melody_pool.extend([x + 12 for x in melody_pool])  # Add octave
            context_melodies = np.random.choice(melody_pool, size=(1, 4, 48))

        else:  # 'real', use real measures from previous bars
            prev_bars = kwargs.get("prev_bars")
            context_rhythms = np.zeros((4, 1, 4))
            context_melodies = np.zeros((1, 4, 48))

            for i, bar in enumerate(prev_bars[-4:]):
                rhythm, melody = self.convertBarToContext(bar)
                context_rhythms[i, :, :] = rhythm
                context_melodies[:, i, :] = melody

        return context_rhythms, context_melodies

    def getLead(self, kwargs: dict, context_rhythms: np.ndarray, context_melodies: np.ndarray) -> list:
        """Returns the last bar of the lead.

        If no lead is given, the last bar of the context is used.
        The user can set the 'lead_mode' in the MusAIc GUI.

        TODO: Add an else statement to return the last bar of the context if the lead is not given.
        TODO: Add option for lead_mode="rhythm" to return only rhythm, use last bar of context for melody.
        """
        if "lead_mode" not in kwargs or not kwargs["lead_mode"]:  # No lead given, use last bar of context
            lead_rhythm = context_rhythms[-1]
            lead_melody = context_melodies[:, -1:, :]

        elif kwargs["lead_mode"] == "both":  # Lead given, return both rhythm and melody
            lead_rhythm, lead_melody = self.convertBarToContext(kwargs["lead_bar"])

        elif kwargs["lead_mode"] == "melody":  # Lead given, return only melody, use last bar of context for rhythm
            lead_rhythm = context_rhythms[-1]
            _, lead_melody = self.convertBarToContext(kwargs["lead_bar"])

        return lead_rhythm, lead_melody

    def sampleOutput(self, output: list, kwargs: dict) -> list:
        """Samples the output of the neural network.

        Either returns the best output (argmax), a weighted sample (dist) or a weighted sample from the top 5 predictions.

        The user can set the 'sample_mode' and 'chord_mode' in the MusAIc GUI.
        """
        sample_mode = kwargs.get("sample_mode", "dist")  # Either "best", "dist", or "top"
        chord_mode = kwargs.get("chord_mode", 1)  # Either "force", "auto", 1, 2, 3, 4

        if chord_mode in ("force", "auto"):
            chord_num = 1
        else:
            chord_num = int(chord_mode)

        if sample_mode in ("argmax", "best"):  # Return the best output
            sampled_rhythm = np.argmax(output[0], axis=-1)
            sampled_melody = np.argmax(output[1], axis=-1)
            sampled_chords = [list(rand.choice(self.vocabulary["melody"], p=curr_p, size=chord_num, replace=True))
                              for curr_p in output[1][0]]  # Sample chord_num chords from the melody distribution

        elif sample_mode == "dist":  # Weighted sample from full distribution
            sampled_rhythm = np.array([[np.random.choice(self.vocabulary["rhythm"], p=dist) for dist in output[0][0]]])
            sampled_melody = np.array([[np.random.choice(self.vocabulary["melody"], p=dist) for dist in output[1][0]]])
            sampled_chords = [list(rand.choice(self.vocabulary["melody"], p=curr_p, size=chord_num, replace=True))
                              for curr_p in output[1][0]]  # Sample chord_num chords from the melody distribution

        elif sample_mode == "top":  # Weighted sample from top 5 predictions
            rhythm = []
            melody = []
            sampled_chords = []

            for i in range(4):  # Sample rhythm
                top5_rhythm_indices = np.argsort(output[0][0][i], axis=-1)[-5:]

                r_probs = output[0][0][i][top5_rhythm_indices]
                r_probs /= sum(r_probs)

                rhythm.append(rand.choice(top5_rhythm_indices, p=r_probs))

            for i in range(len(output[1][0])):  # Sample melody
                top5_m_indices = np.argsort(output[1][0][i], axis=-1)[-5:]
                m_probs = output[1][0][i][top5_m_indices]
                m_probs /= sum(m_probs)

                melody.append(rand.choice(top5_m_indices, p=m_probs))
                sampled_chords.append(list(rand.choice(top5_m_indices, p=m_probs, replace=True, size=chord_num)))

            sampled_rhythm = np.array([rhythm])
            sampled_melody = np.array([melody])

        return sampled_rhythm, sampled_melody, sampled_chords

    def convertBarToContext(self, measure):
        """Converts a list of notes (nn, start_tick, end_tick) to context format for network to use."""
        if not measure or measure.isEmpty():  # Empty bar
            rhythm = [self.rhythmDict[()] for _ in range(4)]
            melody = [random.choice([1, 7]) for _ in range(48)]

            return np.array([rhythm]), np.array([[melody]])

        rhythm = []
        melody = [-1] * 48
        pcs = []

        on_ticks = [False] * 96
        for n in measure.notes:
            try:
                if n[0] <= 0:
                    continue
                on_ticks[n[1]] = True
                melody[n[1] // 2] = n[0] % 12 + 1
                pcs.append(n[0] % 12 + 1)
            except IndexError:
                pass

        for i in range(4):
            beat = on_ticks[i * 24:(i + 1) * 24]
            word = []
            for j in range(24):
                if beat[j]:
                    word.append(round(j / 24, 4))
            try:
                rhythm.append(self.rhythmDict[tuple(word)])
            except KeyError:
                print("[NeuralNet] Beat not found, using eight note...")
                rhythm.append(self.rhythmDict[(0.0, 0.5)])

        if len(pcs) == 0:
            pcs = [1, 8]

        for j in range(48):
            if melody[j] == -1:
                melody[j] = random.choice(pcs)

        return np.array([rhythm]), np.array([[melody]])

    def makeNote(self, pc: Union[int, float], tick_start: int, tick_end: int, octave: int = 4) -> tuple:
        """Convert a pitch class to a note.

        Changes compared to original implementation:
        - Moved function outside convertContextToNotes
        - Added octave parameter, default 4
        """
        nn = 12 * (octave + 1) + pc - 1
        return int(nn), tick_start, tick_end

    def predictChord(self, notes, pc, sample_mode, context_melody, meta_data, chord_mode="auto",
                     octave: int = 4, tick: int = 0, tick_end: int = 96) -> list:
        """Predict a chord using the chordNet.

        Changes compared to original implementation:
        - preprocessMetaData is called here to replace duplicate code
        - Moved function outside convertContextToNotes
        """
        meta_data_processed = self.preprocessMetaData(meta_data)

        model_input = [np.array([[pc]]), np.array([[context_melody]]), meta_data_processed]
        model_output = self.model_chords.predict(x=model_input)

        if sample_mode in ("dist", "top"):
            chord = rand.choice(len(model_output[0]), p=model_output[0])
        else:  # When sample mode is "best"
            chord = np.argmax(model_output[0], axis=-1)

        intervals = self.chordDict[chord]
        if chord_mode == 1:
            intervals = [rand.choice(intervals)]

        for interval in intervals:
            notes.append(self.makeNote(pc + interval - 12, tick, tick_end, octave=octave))

        return notes

    def convertContextToNotes(self, context_rhythm, context_melody, context_chords, kwargs, octave=4) -> list:
        """Convert a context to a list of notes.

        Changes compared to NeuralNet:
        - Provide the chord_mode to predictChord when the chord_mode is "force"
        - Moved functions outside convertContextToNotes
        - Removed the try except block and replaced with if-else to check if we're at the end of the list
        """
        if "meta_data" not in kwargs or kwargs["meta_data"] is None:
            kwargs["meta_data"] = deepcopy(DEFAULT_META_DATA)

        chord_mode = kwargs.get("chord_mode", 1)  # Either "force", "auto", 1, 2, 3, 4
        sample_mode = kwargs.get("sample_mode", "top")  # Either "best", "dist", or "top"

        if chord_mode not in ("force", "auto"):
            chord_mode = int(chord_mode)

        ticks_on = [False] * 96
        for i, beat in enumerate(context_rhythm):
            for onset in self.rhythmDict[beat]:
                ticks_on[int(i + onset) * 24] = True

        ticks_start = [i for i in range(96) if ticks_on[i]]

        notes = []
        for i, tick in enumerate(ticks_start):
            # Removed the try-except block and replaced with if-else
            ticks_end = ticks_start[i + 1] if i + 1 < len(ticks_start) else 96
            pc = context_melody[i // 2]

            if chord_mode == "force":
                tonic = 12 + (pc % 12)

                # Changed chord_mode parameter from none given to force
                notes = self.predictChord(notes, tonic, sample_mode, context_melody, kwargs["meta_data"],
                                          chord_mode=chord_mode, octave=octave, tick=tick, tick_end=ticks_end)

            elif chord_mode in (0, 1, "auto"):
                if pc >= 12:
                    # Draw chord intervals...
                    notes = self.predictChord(notes, pc, sample_mode, context_melody, kwargs["meta_data"],
                                              chord_mode=chord_mode, octave=octave, tick=tick, tick_end=ticks_end)
                else:
                    notes.append(self.makeNote(pc, tick, ticks_end, octave=octave))

            else:  # Only use predicted chords if chord_mode is not {force, auto, 0, 1}
                for chord_pc in context_chords[i // 2]:
                    notes.append(self.makeNote(chord_pc, tick, ticks_end, octave=octave))

        return notes


class TransformerNet(NeuralNet):
    def __init__(self, resources_path: str = None, init_callbacks: list = None) -> None:
        """Initialise the Transformer network.

        Changes compared to NeuralNet:
        - Uses a Transformer instead of an LSTM
        - Changed try-except to if-else for callbacks
        """
        print("[NeuralNet] Initialising...")  # TODO: Change prints to logging
        self.loaded = False  # Set to true when model is loaded
        time_start = time.time()

        print("[NeuralNet] === Using SMT22 model ===")
        self.model = lambda _: 1  # Also needs a .predict(x) method, TODO: implement a model
        self.model_chords = lambda _: 1  # Also needs a .predict(x) method  TODO: either implement seperate model or use the same model

        fp_training_data = "./src/main/resources/base/euroAI/"  # TODO: Change to parameter
        with open(os.path.join(fp_training_data, "DataGenerator.conversion_params"), "rb") as f:
            params_conversion = pickle.load(f)  # TODO: investigate what this is

        with open(os.path.join(fp_training_data, "ChordGenerator.conversion_params"), "rb") as f:
            params_conversion_chord = pickle.load(f)  # TODO: investigate what this is

        self.rhythmDict = params_conversion["rhythm"]
        for k, v in list(self.rhythmDict.items()):  # Reverse dict
            self.rhythmDict[v] = k

        self.chordDict = params_conversion_chord["chords"]
        for k, v in list(self.chordDict.items()):  # Reverse dict
            self.chordDict[v] = k

        self.vocabulary = {"rhythm": 30, "melody": 25}  # Extracted from EuroAI model, TODO: investigate what this is

        self.note_durations_dict = {"qb": [self.rhythmDict[(0.0,)]] * 2,  # Quarter
                                    "lb": [self.rhythmDict[()]],  # Long
                                    "eb": [self.rhythmDict[(0.0, 0.5)], self.rhythmDict[(0.5,)]] * 2,  # Eighth
                                    "fb": [self.rhythmDict[(0.0, 0.25, 0.5, 0.75)],  # Sixteenth
                                           self.rhythmDict[(0.0, 0.25, 0.5)],
                                           self.rhythmDict[(0.5, 0.75)]] * 2,
                                    "tb": [self.rhythmDict[(0.0, 0.333, 0.6667)],  # Triplet
                                           self.rhythmDict[(0.3333, 0.6667)]] * 2, }

        self.scales_dict = {"maj": [1, 3, 5, 6, 8, 10, 12],  # Major
                            "min": [1, 3, 4, 6, 8, 10, 11],  # Minor
                            "pen": [1, 4, 6, 8, 11],  # Pentatonic
                            "5th": [1, 8]}  # 5th

        # Predict some junk data to fully initialise model...
        _ = self.generateBar(**DEFAULT_SECTION_PARAMS, **DEFAULT_AI_PARAMS)  # TODO: investigate if this is necessary

        print(f"[NeuralNet] Neural network loaded in {int(time.time() - time_start)} seconds")

        self.loaded = True  # Set to true when model is loaded

        if init_callbacks:  # Call any callbacks
            if not hasattr(init_callbacks, "__iter__"):  # Check if iterable
                init_callbacks = [init_callbacks]

            for f in init_callbacks:
                f()

    def generateBar(self, octave: int = 4, **kwargs) -> list:
        """Generate a bar of music.

        Currently just random notes for testing.
        In the future this will be replaced with a call to the Transformer model.
        """
        # Preprocess input
        context_rhythms, context_melodies = self.getContexts(kwargs)
        meta_data_preprocessed = self.preprocessMetaData(kwargs["meta_data"])  # Data has not been run through embedding layer yet
        lead_rhythm, lead_melody = self.getLead(kwargs, context_rhythms, context_melodies)

        model_input = [*context_rhythms, context_melodies, meta_data_preprocessed, lead_rhythm, lead_melody]

        # TEMP: Write model input to file for debugging
        with open("model_input.txt", "w+") as f:
            for name, i in zip(["context_rhythms", "context_melodies", "meta_data_embedded", "lead_rhythm", "lead_melody"],
                               [context_rhythms, context_melodies, meta_data_preprocessed, lead_rhythm, lead_melody]):
                f.write(f"{name}:\n{i}\n\n")

        # model_output = self.model.predict(x=model_input)  # TODO: implement model
        # sampled_rhythm, sampled_melody, sampled_chords = self.sampleOutput(model_output, kwargs)  # Postprocess output
        # return_val = self.convertContextToNotes(sampled_rhythm[0], sampled_melody[0], sampled_chords, kwargs, octave=octave)

        # Temporary random notes
        notes = []
        for i in range(4):
            note = (random.randint(60, 80), i * 24, (i + 1) * 24)  # pitch, start tick, end tick
            notes.append(note)

        return notes


class NetworkEngine(multiprocessing.Process):
    def __init__(self, requestQueue, returnQueue, resources_path=None, init_callbacks=None):
        super(NetworkEngine, self).__init__()

        self.requestQueue = requestQueue
        self.returnQueue = returnQueue
        self.resources_path = resources_path
        self.init_callbacks = init_callbacks

        self.stopRequest = multiprocessing.Event()

        self.network = None

    def run(self) -> None:
        """Starts the network engine process.

        Changes from the original:
        - Added support for the TransformerNet
        """
        if not self.network:
            if PLAYER in (VER_9, EUROAI):
                self.network = NeuralNet(resources_path=self.resources_path, init_callbacks=self.init_callbacks)

            elif PLAYER == SMT22:
                self.network = TransformerNet(resources_path=self.resources_path, init_callbacks=self.init_callbacks)

            elif PLAYER == RANDOM:
                self.network = RandomPlayer()

            print("[NetworkEngine] network loaded")

        while not self.stopRequest.is_set():
            try:
                requestMsg = self.requestQueue.get(timeout=1)
                # print("[NetworkEngine]", "request received from", requestMsg["measure_address"])
            except multiprocessing.queues.Empty:
                # print("No messages received yet")
                continue

            # print("Generating result...")
            result = self.network.generateBar(**requestMsg["request"])
            # print("Generated result")

            self.returnQueue.put({"measure_address": requestMsg["measure_address"], "result": result})

            time.sleep(0.01)  # TODO: try to remove this

    def isLoaded(self) -> bool:
        """Returns True if the network is loaded, False otherwise."""
        return self.network.loaded

    def join(self, timeout=1):
        self.stopRequest.set()
        super(NetworkEngine, self).join(timeout)
