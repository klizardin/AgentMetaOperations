from RLParking.settings import *
from RLParking.request import Request, RequestType
from env.vehicle import StaticEnv
from RLParking.db import TrainStates, StartCoordinates, AsyncTrainDBProcessor
from RLParking.bot import AsyncVehicleBot
from RLParking.train import AsyncTrain
from RLParking.asyncnet import AsyncNet


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
    train_db = TrainStates(max_length=GAME_STATES_IN_DB_SIZE_MAX, reward_ratio=GAME_STATES_IN_DB_REWARD_RATIO)

    if PATHES.has_train_state():
        train_db.reduce_size()
        train_db.save(PATHES.get_train_state())

    start_coordinates = StartCoordinates()
    env = StaticEnv()

    async_funcs = {
        AsyncFunc.create_from_class(
            agent_cls=AsyncVehicleBot(ui_bot=False,start_coordinates=start_coordinates, env=env)
        ) : Request(RequestType.NOP),
        AsyncFunc.create_from_class(
            agent_cls=AsyncVehicleBot(ui_bot=True,start_coordinates=start_coordinates, env=env)
        ) : Request(RequestType.NOP),
        AsyncFunc.create_from_class(
            agent_cls=AsyncTrain(db=train_db, env=env)
        ) : Request(RequestType.NOP),
        # AsyncFunc.create_from_class(agent_cls=AsyncCNNTrainDataCreater(db=game_db)) : Request(RequestType.NOP),
    }
    processors = [
        Processor.create_from_class(agent_cls=AsyncNet(db=train_db)),
        Processor.create_from_class(agent_cls=AsyncTrainDBProcessor(db=train_db))
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
            assert (r.is_type(RequestType.NOP))
            r.processed = True
    pass


if __name__ == "__main__":
    main()
