import random
from typing import List, AnyStr

import numpy as np

from game.tetris import OperationInfo, TetrisState, Position, game_metrics
from mc.config import *
from game.BotGame import TetrisPositionInfo

import pickle
import sys
import random


class PlayedState:
    def __init__(self, *, tetris_state : TetrisState,
            start_position : TetrisPositionInfo,
            all_positions: List[TetrisPositionInfo],
            end_positions: List[TetrisPositionInfo],
            illegal_positions: List[TetrisPositionInfo]
            ):
        self.tetris_state = tetris_state
        self.start_position = start_position
        self.all_positions = all_positions
        self.end_positions = end_positions
        self.illegal_positions = illegal_positions
        self.getted = 0

    pass #class PlayedState


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def has_reward(state: PlayedState):
    return any([pt.has_reward() for pt in state.end_positions])


class PlayedGames:
    def __init__(self, max_size : int):
        self._max_size = max_size
        self._states = list()
        self._indexes = list()
        self._avg_getted = np.float32(0.0)
        self._with_reward = None

    def __len__(self):
        return len(self._states)

    def tight_objects(self):
        if len(self._states) <= self._max_size:
            return
        random.shuffle(self._states)
        for s in self._states[self._max_size:]:
            self._avg_getted += (s.getted - self._avg_getted)*np.float32(1e-3)
        self._states = self._states[:self._max_size]
        self._setup_indexes()
        if VERBOSE:
            print("Avg getted from db: {0}".format(self._avg_getted))

    def print_size(self):
        if VERBOSE:
            print("db size = {0}".format(len(self._states))) # " memory size={1}", get_size(self._states)

    def _setup_indexes(self):
        self._indexes = list(range(len(self._states)))
        self._indexes = sorted(self._indexes, key=lambda i:not has_reward(self._states[i]))
        self._with_reward = sum([1 if has_reward(s) else 0 for s in self._states])

    def add_state(self, state: PlayedState):
        self._states.append(state)
        self._indexes.append(len(self._states)-1)

    def get_rand_state(self):
        if not self._states:
            return None
        if not self._with_reward:
            self._setup_indexes()
        if random.random() < max((MIN_REWARDED_STATES_RATE, self._with_reward/len(self._states))):
            ir = random.randint(0,self._with_reward-1)
        else:
            ir = random.randint(self._with_reward, len(self._states)-1)
        s = self._states[self._indexes[ir]]
        s.getted += 1
        return s

    def get_state(self, index: int):
        if index < 0 or not self._states:
            return None
        s = self._states[min((index,len(self._states)-1))]
        s.getted += 1
        return s

    def save(self, fn: AnyStr):
        with open(fn, 'wb') as f:
            pickle.dump(self._states, f)

    def load(self, fn: AnyStr):
        with open(fn, 'rb') as f:
            self._states = pickle.load(f)
        self._setup_indexes()
        self._init_stat()

    def _init_stat(self):
        reward = 0
        for s in self._states:
            reward += max([pos.get_reward() for pos in s.end_positions])
        reward /= len(self._states)
        game_metrics.initialize(reward)

    def clear(self):
        self._states = list()


    pass #class PlayedGames

