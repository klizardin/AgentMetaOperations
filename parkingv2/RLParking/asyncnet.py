from DNN.net import NetData, NetTrainData, RLModel
from RLParking.request import RequestType, Request
from RLParking.db import TrainStates
import RLParking.settings as settings
import time
import tensorflow as tf
import numpy as np


class AsyncNet:
    def __init__(self, db: TrainStates):
        self._can_process_requests = [
            RequestType.GET_BEST_OPERATION,
            RequestType.SEARCH_OPERATION,
            RequestType.RL_TRAIN
        ]
        self._init_model()
        self._last_save_t = time.time()
        self._db = db

    def _init_model(self):
        self._model = RLModel()
        self._model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)
            , loss = 'mse'
            , metrics = ['mse']
            )

        if settings.PATHES.has_main_model():
            # https://github.com/tensorflow/tensorflow/issues/24623
            input_size = settings.NET_INPUT_SIZE + settings.OPERATIONS_COUNT*settings.NET_OPERATION_ITEM_SIZE
            x = np.zeros((settings.STATES_TO_TRAIN_BATCH_SIZE, input_size), dtype=np.float32)
            y = np.zeros((settings.STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
            self._model.fit(
                x, y,
                epochs=1, batch_size=settings.STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
                callbacks=None
            )

        if settings.PATHES.has_main_model():
            self._model.load_weights(settings.PATHES.get_main_model())
            print("model loaded {0}".format(settings.PATHES.get_main_model()))
        #    coef1 = tf.Variable(settings.RESET_WEIGHTS_COEF.RESET_WEIGHTS_COEF)
        #    tf.keras.backend.get_session().run(coef1.initializer)
        #    for l in self._model.layers_p1:
        #        for w in l.trainable_weights:
        #            assign1 = w.assign(w*coef1)
        #            tf.keras.backend.get_session().run(assign1)
        #    for l in self._model.layers_p2:
        #        for w in l.trainable_weights:
        #            assign1 = w.assign(w*coef1)
        #            tf.keras.backend.get_session().run(assign1)

    def _get_values(self, nd: NetData):
        return self._model.predict(nd.x, nd.x.shape[0]) #- np.float32(0.5)

    def _train(self, ntd: NetTrainData, batch_size):
        t = time.time()
        callbacks = []
        if t > self._last_save_t + settings.MAIN_MODEL_SAVE_INTERVAL:
            self._model.save_weights(filepath=settings.PATHES.get_main_model(), overwrite=True, save_format="h5")
            self._last_save_t = t
            if settings.VERBOSE_MAIN_MODEL_SAVE:
                print("saved model {0}".format(settings.PATHES.get_main_model()))
        return self._model.fit(
            ntd.x, ntd.y,
            epochs=1, batch_size=batch_size, verbose=0,
            callbacks = callbacks if callbacks else None,
            use_multiprocessing=True,
            workers=2,
            )

    @staticmethod
    def _get_net_data_for_run(inputs, ops):
        inputs_np = np.array(inputs, dtype=np.float32).flatten()
        x = list()
        for op in ops:
            op_arr = np.full((settings.OPERATIONS_COUNT,), np.float32(-0.5), dtype=np.float32)
            op_arr[int((op[0] + 0.5)*settings.ANGLES_FOR_OPERATIONS)] = np.float32(0.5)
            op_arr[settings.ANGLES_FOR_OPERATIONS+0] = np.fabs(op[1])
            op_arr[settings.ANGLES_FOR_OPERATIONS+1] = 0.5 if op[1] >= np.float32(0.0) else -0.5
            # print("input: {}  {}".format(int((op[0] + 0.5)*settings.ANGLES_FOR_OPERATIONS), op[1]))
            x.append(np.concatenate((inputs_np, op_arr)))
        return NetData(np.stack(x))

    @staticmethod
    def _fill_result(request: Request, nd: NetData):
        request.results = nd.y.flatten()

    def _set_search_operation(self, nd: NetData):
        can_train = self._db.can_train()
        if ((not can_train and (np.random.rand() > settings.RL_PRETRAIN_SEARCH_COEF))
               or (can_train and (np.random.rand() > settings.RL_SEARCH_COEF))
            ):
            return
        if settings.RL_SEARCH_USE_ALPHA_ALGORITHM or (not can_train):
            v = np.random.randint(0,nd.y.shape[0],size=1, dtype=np.int32)
        else:
            p = nd.y.flatten()
            p -= np.min(p)
            s = np.sum(p)
            if s > 0:
                p /= s
            else:
                p[:] = np.float32(1.0)/p.shape[0]
            v = np.random.choice(nd.y.shape[0],1,p=p)
        nd.y[v[0], :] = np.float32(settings.VALUE_MORE_THAN_MAX_NET_RESULT)

    @staticmethod
    def _get_train_data(inputs, ops, values):
        x = list()
        for input1, op in zip(inputs,ops):
            op_arr = np.full((settings.OPERATIONS_COUNT,),np.float32(-0.5),dtype=np.float32)
            op_arr[int((op[0] + 0.5) * settings.ANGLES_FOR_OPERATIONS)] = np.float32(0.5)
            op_arr[settings.ANGLES_FOR_OPERATIONS+0] = np.fabs(op[1])
            op_arr[settings.ANGLES_FOR_OPERATIONS+1] = 0.5 if op[1] >= np.float32(0.0) else -0.5
            input1_np = np.array(input1, dtype=np.float32).flatten()
            x.append(np.concatenate((input1_np,op_arr)))
        assert(len(x)==values.shape[0])
        return NetTrainData(np.stack(x), values)

    def process(self, request: Request):
        if not request.in_types(self._can_process_requests):
            return False
        if request.is_type(RequestType.GET_BEST_OPERATION):
            net_data = AsyncNet._get_net_data_for_run(request.inputs, request.ops)
            net_data.y = self._get_values(net_data)
            self._fill_result(request, net_data)
        elif request.is_type(RequestType.SEARCH_OPERATION):
            net_data = AsyncNet._get_net_data_for_run(request.inputs, request.ops)
            net_data.y = self._get_values(net_data)
            self._set_search_operation(net_data)
            AsyncNet._fill_result(request, net_data)
        elif request.is_type(RequestType.RL_TRAIN):
            net_train_data = AsyncNet._get_train_data(request.inputs, request.ops, request.values)
            self._train(net_train_data, settings.STATES_TO_TRAIN_BATCH_SIZE)
            pass
        return True

    pass # class AsyncNet

