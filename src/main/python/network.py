
#pylint: disable=invalid-name,missing-docstring

import os
import time
import random
import multiprocessing
from copy import deepcopy

import pickle as pkl

import numpy as np
import numpy.random as rand

from core import DEFAULT_SECTION_PARAMS, DEFAULT_AI_PARAMS, DEFAULT_META_DATA

RANDOM = 0
VER_9 = 1
EUROAI = 2

PLAYER = 2

if PLAYER != RANDOM:
    from v9.Nets.ChordNetwork import ChordNetwork
    from v9.Nets.MetaEmbeddingEuro import MetaEmbedding
    from v9.Nets.MetaPredictorEuro import MetaPredictor
    from v9.Nets.CombinedNetworkEuro import CombinedNetwork


class RandomPlayer():
    ''' For testing purpose only! '''

    def __init__(self):
        print('[RandomPlayer]', ' === Using RANDOM PLAYER for testing ===')

    def generateBar(self, **kwargs):
        notes = []
        for i in range(4):
            note = (random.randint(60, 80), i*24, (i+1)*24)
            notes.append(note)

        return notes


class NeuralNet():

    def __init__(self, resources_path=None, init_callbacks=None):

        print('[NeuralNet]', 'Initialising...')
        self.loaded = False

        startTime = time.time()

        if not resources_path:
            resources_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/base/')

        if PLAYER == VER_9:
            trainingsDir = "./src/main/resources/base/v9_lead/"
        elif PLAYER == EUROAI:
            trainingsDir = "./src/main/resources/base/euroAI/"
        else:
            raise ('[NeuralNet] Unknown player initialised ({}). Aborting'.format(PLAYER))

        print('\n[NeuralNet]', ' === Using {} ===\n'.format('VER9' if PLAYER == VER_9 else 'EUROAI'))

        with open(os.path.join(trainingsDir, 'DataGenerator.conversion_params'), 'rb') as f:
            conversionParams = pkl.load(f)

        self.rhythmDict = conversionParams['rhythm']
        for k, v in list(self.rhythmDict.items()):
            self.rhythmDict[v] = k

        self.metaEmbedder = MetaEmbedding.from_saved_custom(os.path.join(trainingsDir, 'meta'))
        metaPredictor = MetaPredictor.from_saved_custom(os.path.join(trainingsDir, 'meta'))

        weightsFolder = os.path.join(trainingsDir, 'weights')
        self.combinedNet = CombinedNetwork.from_saved_custom(weightsFolder, metaPredictor,
                                                             generation=True, compile_now=False)

        self.vocabulary = {
            'rhythm': self.combinedNet.params['rhythm_net_params'][2],
            'melody': self.combinedNet.params['melody_net_params'][3]
        }

        with open(os.path.join(trainingsDir, 'ChordGenerator.conversion_params'), 'rb') as f:
            chordConversionParams = pkl.load(f)

        self.chordDict = chordConversionParams['chords']
        for k, v in list(self.chordDict.items()):
            self.chordDict[v] = k

        self.chordNet = ChordNetwork.from_saved_custom(os.path.join(trainingsDir, 'chord'),
                                                       load_melody_encoder=True)

        # predict some junk data to fully initilise model...
        self.generateBar(**DEFAULT_SECTION_PARAMS, **DEFAULT_AI_PARAMS)

        print('\n[NeuralNet]', 'Neural network loaded in', int(time.time() - startTime), 'seconds\n')

        self.loaded = True

        if init_callbacks:
            try:
                for f in init_callbacks:
                    f()
            except:
                init_callbacks()

    def generateBar(self, octave=4, **kwargs):
        ''' Expecting...
            - 'lead_bar'
            - 'prev_bars'
            - 'sample_mode'
            - 'chord_mode'
            - 'lead_mode'
            - 'context_mode'
            - 'injection_params'
            - 'meta_data'
            - 'octave'
        '''

        #print('[NeuralNet]', 'generateBar with params')
        # for k, v in kwargs.items():
        #    print('  ', k, v)

        rhythmContexts, melodyContexts = self.getContexts(kwargs)
        embeddedMetaData = self.embedMetaData(kwargs['meta_data'])
        leadRhythm, leadMelody = self.getLead(kwargs, rhythmContexts, melodyContexts)

        output = self.combinedNet.predict(x=[*rhythmContexts,
                                             melodyContexts,
                                             embeddedMetaData,
                                             leadRhythm,
                                             leadMelody])

        sampledRhythm, sampledMelody, sampledChords = self.sampleOutput(output, kwargs)

        return self.convertContextToNotes(sampledRhythm[0],
                                          sampledMelody[0],
                                          sampledChords,
                                          kwargs,
                                          octave=octave)

    def embedMetaData(self, metaData):
        if not metaData:
            metaData = DEFAULT_META_DATA
        values = []
        #print('[NeuralNet]', 'embedMetaData:')
        for k in sorted(metaData.keys()):
            #print(k, metaData[k])
            if k == 'ts':
                values.extend([4, 4])
            else:
                values.append(metaData[k])

        md = np.tile(values, (1, 1))

        return self.metaEmbedder.predict(md)

    def getContexts(self, kwargs):
        mode = kwargs.get('context_mode', None)
        injection_params = kwargs.get('injection_params',
                                      DEFAULT_AI_PARAMS['injection_params'])
        prev_bars = kwargs.get('prev_bars')

        if mode == 'inject':
            rhythmPool = []
            for rhythmType in injection_params[0]:
                # *2 gives extra weight to non-empty beats
                rhythmPool.extend({
                    'qb': [self.rhythmDict[(0.0,)]]*2,
                    'lb': [self.rhythmDict[()]],
                    'eb': [self.rhythmDict[(0.0, 0.5)],
                           self.rhythmDict[(0.5,)]]*2,
                    'fb': [self.rhythmDict[(0.0, 0.25, 0.5, 0.75)],
                           self.rhythmDict[(0.0, 0.25, 0.5)],
                           self.rhythmDict[(0.5, 0.75)]]*2,
                    'tb': [self.rhythmDict[(0.0, 0.333, 0.6667)],
                           self.rhythmDict[(0.3333, 0.6667)]]*2,
                }[rhythmType])

            rhythmContexts = [np.random.choice(rhythmPool, size=(1, 4)) for _ in range(4)]

            melodyPool = {
                'maj': [1, 3, 5, 6, 8, 10, 12],
                'min': [1, 3, 4, 6, 8, 10, 11],
                'pen': [1, 4, 6, 8, 11],
                '5th': [1, 8]
            }[injection_params[1]]
            # if len(injection_params) > 2 and injection_params[2]:
            melodyPool.extend([x+12 for x in melodyPool])

            melodyContexts = np.random.choice(melodyPool, size=(1, 4, 48))

        else:
            rhythmContexts = np.zeros((4, 1, 4))
            melodyContexts = np.zeros((1, 4, 48))
            for i, b in enumerate(prev_bars[-4:]):
                r, m = self.convertBarToContext(b)
                rhythmContexts[i, :, :] = r
                melodyContexts[:, i, :] = m

        #print('[NeuralNet]', 'Contexts:', rhythmContexts, melodyContexts)

        return rhythmContexts, melodyContexts

    def getLead(self, kwargs, rhythmContexts, melodyContexts):
        if 'lead_mode' not in kwargs or not kwargs['lead_mode']:
            leadRhythm = rhythmContexts[-1]
            leadMelody = melodyContexts[:, -1:, :]
        elif kwargs['lead_mode'] == 'both':
            leadRhythm, leadMelody = self.convertBarToContext(kwargs['lead_bar'])
        elif kwargs['lead_mode'] == 'melody':
            leadRhythm = rhythmContexts[-1]
            _, leadMelody = self.convertBarToContext(kwargs['lead_bar'])

        return leadRhythm, leadMelody

    def sampleOutput(self, output, kwargs):
        mode = kwargs.get('sample_mode', 'dist')
        chord_mode = kwargs.get('chord_mode', 1)
        if chord_mode in {'force', 'auto'}:
            chord_num = 1
        else:
            chord_num = int(chord_mode)

        #print('[NeuralNet]', 'sampleOutput', 'chord_mode', chord_mode, 'chord_num', chord_num)

        if mode == 'argmax' or mode == 'best':
            sampledRhythm = np.argmax(output[0], axis=-1)
            sampledMelody = np.argmax(output[1], axis=-1)
            sampledChords = [list(rand.choice(self.vocabulary['melody'], p=curr_p,
                                              size=chord_num, replace=True)) for curr_p in output[1][0]]
        elif mode == 'dist':
            sampledRhythm = np.array([[np.random.choice(self.vocabulary['rhythm'], p=dist)
                                       for dist in output[0][0]]])
            sampledMelody = np.array([[np.random.choice(self.vocabulary['melody'], p=dist)
                                       for dist in output[1][0]]])
            sampledChords = [list(rand.choice(self.vocabulary['melody'], p=curr_p, size=chord_num,
                                              replace=True)) for curr_p in output[1][0]]
        elif mode == 'top':
            # Random from top 5 predictions....
            r = []
            sampledChords = []
            for i in range(4):
                top5_rhythm_indices = np.argsort(output[0][0][i], axis=-1)[-5:]

                r_probs = output[0][0][i][top5_rhythm_indices]
                r_probs /= sum(r_probs)

                r.append(rand.choice(top5_rhythm_indices, p=r_probs))

            sampledRhythm = np.array([r])
            m = []

            for i in range(len(output[1][0])):
                top5_m_indices = np.argsort(output[1][0][i], axis=-1)[-5:]
                m_probs = output[1][0][i][top5_m_indices]
                m_probs /= sum(m_probs)

                m.append(rand.choice(top5_m_indices, p=m_probs))
                sampledChords.append(list(rand.choice(top5_m_indices, p=m_probs,
                                                      replace=True, size=chord_num)))
            sampledMelody = np.array([m])

        #print('[NeuralNet]', sampledRhythm.shape, sampledMelody.shape)
        return sampledRhythm, sampledMelody, sampledChords

    def convertBarToContext(self, measure):
        '''
        Converts a list of notes (nn, start_tick, end_tick) to context
        format for network to use
        '''

        if not measure or measure.isEmpty():
            # empty bar...
            rhythm = [self.rhythmDict[()] for _ in range(4)]
            melody = [random.choice([1, 7]) for _ in range(48)]
            return np.array([rhythm]), np.array([[melody]])

        # print(measure.notes)

        rhythm = []
        melody = [-1]*48
        pcs = []

        onTicks = [False] * 96
        for n in measure.notes:
            try:
                if n[0] <= 0:
                    continue
                onTicks[n[1]] = True
                melody[n[1]//2] = n[0] % 12 + 1
                pcs.append(n[0] % 12+1)
            except IndexError:
                pass

        for i in range(4):
            beat = onTicks[i*24:(i+1)*24]
            word = []
            for j in range(24):
                if beat[j]:
                    word.append(round(j/24, 4))
            try:
                rhythm.append(self.rhythmDict[tuple(word)])
            except KeyError:
                print('[NeuralNet]', 'Beat not found, using eigth note...')
                rhythm.append(self.rhythmDict[(0.0, 0.5)])

        if len(pcs) == 0:
            pcs = [1, 8]

        for j in range(48):
            if melody[j] == -1:
                melody[j] = random.choice(pcs)

        return np.array([rhythm]), np.array([[melody]])

    def convertContextToNotes(self, rhythmContext, melodyContext,
                              chordContexts, kwargs, octave=4):

        def makeNote(pc, startTick, endTick):
            nn = 12*(octave+1) + pc - 1
            note = (int(nn), startTick, endTick)
            return note

        def predictChord(notes, pc, sample_mode, melodyContext, metaData, chord_mode='auto'):
            values = []
            for k in sorted(metaData.keys()):
                if k == 'ts':
                    values.extend([4, 4])
                else:
                    values.append(metaData[k])
            md = np.tile(values, (1, 1))

            chord_outputs = self.chordNet.predict(
                x=[np.array([[pc]]), np.array([[melodyContext]]), md]
            )
            if sample_mode == 'dist' or sample_mode == 'top':
                chord = rand.choice(len(chord_outputs[0]), p=chord_outputs[0])
            else:
                chord = np.argmax(chord_outputs[0], axis=-1)

            intervals = self.chordDict[chord]
            if chord_mode == 1:
                intervals = [rand.choice(intervals)]
            for interval in intervals:
                notes.append(makeNote(pc+interval-12, tick, endTick))

            return notes

        if 'meta_data' not in kwargs or kwargs['meta_data'] == None:
            kwargs['meta_data'] = deepcopy(DEFAULT_META_DATA)

        notes = []
        onTicks = [False] * 96

        chord_mode = kwargs.get('chord_mode', 1)
        if chord_mode not in {'force', 'auto'}:
            chord_mode = int(chord_mode)

        #print('[NeuralNet]', 'convertContextToNotes', 'chord_mode', chord_mode)
        sample_mode = kwargs.get('sample_mode', 'top')

        for i, beat in enumerate(rhythmContext):
            b = self.rhythmDict[beat]
            for onset in b:
                onTicks[int((i+onset)*24)] = True

        startTicks = [i for i in range(96) if onTicks[i]]

        for i, tick in enumerate(startTicks):
            try:
                endTick = startTicks[i+1]
            except:
                endTick = 96
            pc = melodyContext[i//2]

            if chord_mode == 'force':
                tonic = 12 + (pc % 12)
                notes = predictChord(notes, tonic, sample_mode, melodyContext, kwargs['meta_data'])
            elif chord_mode in (0, 1, 'auto'):
                if pc >= 12:
                    # draw chord intervals...
                    notes = predictChord(notes, pc, sample_mode, melodyContext, kwargs['meta_data'], chord_mode=chord_mode)
                else:
                    notes.append(makeNote(pc, tick, endTick))

            else:
                for chord_pc in chordContexts[i//2]:
                    notes.append(makeNote(chord_pc, tick, endTick))

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

    def run(self):
        if not self.network:
            if PLAYER == VER_9 or PLAYER == EUROAI:
                self.network = NeuralNet(resources_path=self.resources_path,
                                         init_callbacks=self.init_callbacks)
            elif PLAYER == RANDOM:
                self.network = RandomPlayer()

            print('[NetworkEngine]', 'network loaded')

        while not self.stopRequest.is_set():
            try:
                requestMsg = self.requestQueue.get(timeout=1)
                #print('[NetworkEngine]', 'request recieved from', requestMsg['measure_address'])
            except multiprocessing.queues.Empty:
                #print('no messages recieved yet')
                continue

            #print('generating result...')
            result = self.network.generateBar(**requestMsg['request'])
            #print('generated result')

            self.returnQueue.put({'measure_address': requestMsg['measure_address'],
                                  'result': result})

            time.sleep(0.01)

    def isLoaded(self):
        return self.network.loaded

    def join(self, timeout=1):
        self.stopRequest.set()
        super(NetworkEngine, self).join(timeout)


# EOF
