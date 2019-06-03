from enum import Enum
from typing import List

import numpy as np

from game.tetris import FieldState


class RequestType(Enum):
    NOP = 0                         # no need to react for this event
    GET_BEST_OPERATION = 1         # need to return
    SEARCH_BEST_OPERATION = 2
    GET_BEST_OPERATION_BATCH = 6
    TRAIN = 7
    SET_TRAIN_DATA = 8

    pass #class RequestType


class Request:
    def __init__(self,
                 type : RequestType, *,
                 field_state : FieldState = None,
                 field_states : List[FieldState] = None,
                 tetris_state = None,
                 start_position = None,
                 all_positions = None,
                 end_positions = None,
                 illegal_positions = None
                 ):
        self.type = type
        self.field_state = field_state
        self.field_states = field_states
        self.tetris_state = tetris_state
        self.start_position = start_position
        self.all_positions = all_positions
        self.end_positions = end_positions
        self.illegal_positions = illegal_positions

        self._check_state()
        self.processed = False

    def _check_state(self):
        assert(isinstance(self.type,RequestType))
        if self.type.value == RequestType.NOP.value:
            return True
        elif (
                (self.type.value == RequestType.GET_BEST_OPERATION.value)
                or (self.type.value == RequestType.SEARCH_BEST_OPERATION.value)
            ):
            assert (self.field_state is not None)
            assert (self.field_state.ops is not None)
            return True
        elif (
                (self.type.value == RequestType.GET_BEST_OPERATION_BATCH.value)
                or (self.type.value == RequestType.TRAIN.value)
            ):
            assert (self.field_states is not None)
            return True
        elif self.type.value == RequestType.SET_TRAIN_DATA.value:
            assert(self.tetris_state is not None)
            assert(self.start_position is not None)
            #assert(self.all_positions is not None)
            assert(self.end_positions is not None)
            assert(self.illegal_positions is not None)
            return True
        return True

    pass #class Request
