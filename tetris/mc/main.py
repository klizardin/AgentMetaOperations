import time

import numpy as np
import tensorflow as tf

from game.BotGame import BotGame
from mc.DBAsync import DBAsync, AsyncCNNTrainDataCreater
from mc.config import *
from mc.request import RequestType, Request
from nn.PlayedDB import PlayedGames, PlayedState
from nn.net import RLModel, CNNPreTrainModel
import os


class RLProcessor:
    def __init__(self):
        self._model = RLModel(width=FIELD_WIDTH, height=FIELD_FULL_HEIGHT)
        self._model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
            , loss = 'mse'
            , metrics = ['mse']
            )

        # https://github.com/tensorflow/tensorflow/issues/24623
        x = np.ones((STATES_TO_TRAIN_BATCH_SIZE, FIELD_WIDTH * FIELD_FULL_HEIGHT * 2), dtype=np.float32)
        y = np.ones((STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
        self._model.fit(
            x, y,
            epochs=1, batch_size=STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
            callbacks=None
        )

        if os.path.isfile(MODEL_WEIGHTS_PATH):
            #self._model.print_weights()
            w1 = np.copy(self._model.cnn_layers[0].get_weights())
            self._model.load_weights(MODEL_WEIGHTS_PATH)
            w2 = np.copy(self._model.cnn_layers[0].get_weights())
            assert (not (w1 == w2))
            print("model loaded {0}".format(MODEL_WEIGHTS_PATH))
            #self._model.print_weights()
            self._model.freeze_cnn_weights()
        elif os.path.isfile(CNNMODEL_BEST_WEIGHTS_PATH):
            output_size = (
                    FIELD_WIDTH * FIELD_FULL_HEIGHT  # field to reconstruct
                    + 1  # is legal operation
                    + 5  # operations possible
                    + 4  # does get any reward
                    + 1  # dot fill
            )
            cnnmodel = CNNPreTrainModel(width=FIELD_WIDTH, height=FIELD_FULL_HEIGHT, output_size=output_size)
            cnnmodel.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
                          , loss='mse'
                          , metrics=['mse']
                          )

            x = np.ones((STATES_TO_TRAIN_BATCH_SIZE, FIELD_WIDTH * FIELD_FULL_HEIGHT * 2), dtype=np.float32)
            y = np.ones((STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
            cnnmodel.fit(
                x, y,
                epochs=1, batch_size=STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
                callbacks=None
            )
            w1 = np.copy(cnnmodel.cnn_layers[0].get_weights())
            cnnmodel.load_weights(CNNMODEL_BEST_WEIGHTS_PATH)
            w2 = np.copy(cnnmodel.cnn_layers[0].get_weights())
            assert(not (w1 == w2))

            self._model.copy_weights_from(cnnmodel)
            self._model.freeze_cnn_weights()
            print("copied weightes from cnn pre train model {0}".format(CNNMODEL_BEST_WEIGHTS_PATH))

        self._model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
            , loss = 'mse'
            , metrics = ['accuracy']
            )

        self._last_save_t = time.time()

    def _get_values(self, x):
        x = x - np.float32(0.5)
        return self._model.predict(x, x.shape[0]) #- np.float32(0.5)

    def _train(self, *, x, y, batch_size):
        x = x - np.float32(0.5)
        #y = y + np.float32(0.5)
        t = time.time()
        callbacks = []
        if t > self._last_save_t + 60*25:
            self._model.save_weights(filepath=MODEL_WEIGHTS_PATH, overwrite=True, save_format="h5")
            self._last_save_t = t
            if VERBOSE:
                print("saved model {0}".format(MODEL_WEIGHTS_PATH))
        return self._model.fit(
            x,y,
            epochs=1, batch_size=batch_size, verbose=0,
            callbacks = callbacks if callbacks else None
            )

    def _prepare_input(self, request : Request):
        field = request.field_state.field.flatten()
        batch = [np.concatenate((field, op.op_arr.flatten())) for op in request.field_state.ops]
        return np.stack(batch)

    def _pack_result(self, request : Request, result):
        assert(len(request.field_state.ops) == len(result))
        for i in range(result.shape[0]):
            request.field_state.ops[i].value = result[i]
        pass

    def _prepare_input_batch(self, request : Request):
        batch = list()
        for s in request.field_states:
            field = s.field.flatten()
            batch.extend([np.concatenate((field, op.op_arr.flatten())) for op in s.ops])
        return np.stack(batch)

    def _pack_result_batch(self, request : Request, result):
        i = 0
        for s in request.field_states:
            for j in range(len(s.ops)):
                s.ops[j].value = result[i]
                i += 1
        pass

    def _prepare_input_train(self, request : Request):
        batch_x = list()
        batch_y = list()
        for s in request.field_states:
            field = s.field.flatten()
            batch_x.append(np.concatenate((field, s.op.op_arr.flatten())))
            batch_y.append(np.copy(s.value))
        return np.stack(batch_x), np.stack(batch_y)

    def process(self, request : Request):
        if request.type.value == RequestType.GET_BEST_OPERATION.value:
            if not request.field_state.ops:
                return True
            input = self._prepare_input(request)
            result = self._get_values(input)
            request.field_state.search_step = False
            self._pack_result(request, result)
            return True
        elif request.type.value == RequestType.GET_BEST_OPERATION_BATCH.value:
            if not request.field_states:
                return True
            input = self._prepare_input_batch(request)
            result = self._get_values(input)
            self._pack_result_batch(request, result)
            return True
        elif request.type.value == RequestType.SEARCH_BEST_OPERATION.value:
            if not request.field_state.ops:
                return True
            input = self._prepare_input(request)
            result = self._get_values(input)
            if np.random.ranf() <= RL_SEARCH_COEF:
                request.field_state.search_step = True
                result[np.random.randint(0,len(result))] = np.float32(1e3)
            else:
                request.field_state.search_step = False
            self._pack_result(request, result)
            return True
        elif request.type.value == RequestType.TRAIN.value:
            self._process_train(request)
            return True
        return False

    def _process_train(self, request):
        x, y = self._prepare_input_train(request)
        self._train(x=x, y=y, batch_size=STATES_TO_TRAIN_BATCH_SIZE)

    pass #class RLProcessor

class AsyncFunc:
    def __init__(self, agent_cls = None, agent_func = None):
        self._obj = agent_cls
        self._func = agent_func
        self._it = None

    def init(self):
        if self._obj is not None:
            self._it = self._obj.process()
        if self._func is not None:
            self._it = self._func()
        assert(self._it is not None)
        return next(self._it)

    def process(self, request_result : Request):
        return self._it.send(request_result)

    @classmethod
    def create_from_class(cls, agent_cls):
        return cls(agent_cls = agent_cls)

    @classmethod
    def create_from_func(cls, agent_func):
        return cls(agent_func = agent_func)

    pass #class AsyncFunc

class Processor:
    def __init__(self, agent_cls = None, agent_func = None):
        self._obj = agent_cls
        self._func = agent_func

    def process(self, request : Request):
        if request.processed:
            return
        if self._obj is not None:
            processed = self._obj.process(request)
            request.processed = processed
        if self._func is not None:
            processed = self._func(request)
            request.processed = processed

    @classmethod
    def create_from_class(cls, agent_cls):
        return cls(agent_cls = agent_cls)

    @classmethod
    def create_from_func(cls, agent_func):
        return cls(agent_func = agent_func)

    pass #class Processor


def main():
    game_db = PlayedGames(max_size=GAME_STATES_DB_SIZE_MAX)

    def db_saver(request : Request):
        if request.type.value == RequestType.SET_TRAIN_DATA.value:
            game_db.add_state(PlayedState(
                    tetris_state=request.tetris_state,
                    start_position=request.start_position,
                    all_positions=request.all_positions,
                    end_positions=request.end_positions,
                    illegal_positions=request.illegal_positions
                )
            )
            return True
        return False

    if os.path.isfile(GAME_STATES_PATH):
        game_db.load(GAME_STATES_PATH)
        game_db.tight_objects()
        game_db.save(GAME_STATES_PATH)

    async_funcs = {
        AsyncFunc.create_from_class(agent_cls=BotGame()) : Request(RequestType.NOP),
        AsyncFunc.create_from_class(agent_cls=BotGame(ui_mode=True)) : Request(RequestType.NOP),
        AsyncFunc.create_from_class(agent_cls=DBAsync(db=game_db)) : Request(RequestType.NOP),
        # AsyncFunc.create_from_class(agent_cls=AsyncCNNTrainDataCreater(db=game_db)) : Request(RequestType.NOP),
    }
    processors = [
        Processor.create_from_class(agent_cls=RLProcessor()),
        Processor.create_from_func(agent_func=db_saver)
    ]

    for f in async_funcs.keys():
        async_funcs[f] = f.init()

    while True:
        main_cycle(async_funcs, processors)
    pass

def main_cycle(async_funcs, processors):
    for f, r in async_funcs.items():
        if async_funcs[f].processed:
            async_funcs[f] = f.process(r)
        pass
    for p in processors:
        for r in async_funcs.values():
            if not r.processed:
                p.process(r)
    for r in async_funcs.values():
        if not r.processed:
            assert (r.type.value == RequestType.NOP.value)
            r.processed = True
    pass


if __name__ == "__main__":
    main()

