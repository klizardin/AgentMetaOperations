from enum import Enum
from typing import List
import numpy as np


class RequestType(Enum):
    NOP = 0
    SAVE_TRAIN_STATE = 1
    GET_BEST_OPERATION = 2
    SEARCH_OPERATION = 3
    RL_TRAIN = 4

    pass  # class RequestType


class Request:
    def __init__(self,
        type: RequestType,
        inputs=None,
        ops=None,
        state=None,
        next_state=None,
        reward=None,
        values=None,
        final_state=None,
        ):
        self.type = type
        self.inputs = inputs
        self.ops = ops
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.ops = ops
        self.final_state = final_state
        self.values = values
        self.results = None
        self._check()
        self.processed = False

    def _check(self):
        assert(self.type is not None)
        if self.is_type(RequestType.GET_BEST_OPERATION) or self.is_type(RequestType.SEARCH_OPERATION):
            assert(self.inputs is not None)
            assert(self.ops is not None)
        elif self.is_type(RequestType.SAVE_TRAIN_STATE):
            assert(self.state is not None)
            assert(self.next_state is not None)
            assert(self.reward is not None)
            assert(self.final_state is not None)
        elif self.is_type(RequestType.RL_TRAIN):
            assert(self.inputs is not None)
            assert(self.ops is not None)
            assert(self.values is not None)

    def is_type(self, type: RequestType):
        return self.type.value == type.value

    def in_types(self, types: List[RequestType]):
        return any([type.value == self.type.value for type in types])

    def get_best_operation(self):
        return np.argmax(self.results)

    def get_best_operation_value(self):
        return np.max(self.results)

    pass  # class Request
