from geom.primitives import Point
from env.vehicle import VehicleState
from RLParking.settings import settings
from RLParking.request import RequestType, Request
from geom.utils import radix_sorted, radix_sorted_indexes, froze_class

from typing import AnyStr
import random
import pickle
import numpy as np
import math
from functools import partial


class StartCoord:
    def __init__(self, pos: Point, angle: np.float32, weight: int):
        self.pos = pos
        self.angle = angle
        self.weight = weight
        pass

    pass  # class StartCoord


class StartCoordinates:
    def __init__(self):
        self._start_positions = settings.START_POSITIONS

    def get_rand(self, count: int = 1):
        r = np.random.rand(2)
        p1 = Point.between(self._start_positions[0].pt0, self._start_positions[0].pt1, r[0])
        p2 = Point.between(self._start_positions[1].pt0, self._start_positions[1].pt1, r[0])
        p = Point.between(p1, p2, r[1])
        return [StartCoord(
            pos=p,
            angle=np.random.randint(0, 2)*np.float32(math.pi) + np.float32(math.pi*0.5),
            weight=1
            )
            for _ in range(count)]

    def get_pos_with_weights(self):
        return

    pass  # class StartCoordinates


@froze_class
class TrainState:
    def __init__(self,
            state: VehicleState, next_state: VehicleState,
            reward: np.float32, ops, values,
            final_state: bool
        ):
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.final_state = final_state
        self.ops = ops
        self.values = values
        self.getted = 0

    def has_reward(self):
        if settings.USE_LEARNING_BOOST:
            return self.reward >= settings.REINFORCE_DONE * np.float32(0.8)
        else:
            return ((self.reward >= settings.REINFORCE_DONE * np.float32(0.8))
                or (self.reward <= settings.REINFORCE_FAIL * np.float32(0.8)))

    def get_reward_index(self, max_reward):
        if self.has_values():
            return 3
        elif self.reward >= max_reward * np.float32(0.8):
            return 2
        elif self.reward <= settings.REINFORCE_FAIL * np.float32(0.8):
            return 1
        else:
            return 0

    def has_values(self):
        return self.ops is not None if self.getted < settings.GAME_STATE_GETTED_MAX_COUNT else False

    pass  # class TrainState


class TrainStates:
    def __init__(self, max_length: int, reward_ratio: np.float32):
        self._states = list()
        self._max_length = max_length
        self._reward_ratio = reward_ratio
        if settings.PATHES.has_train_state():
            self._load(settings.PATHES.get_train_state())
        self._init_indexes()
        self._calc_avg_values()

    def __len__(self):
        return len(self._states)

    def can_train(self):
        return len(self._states) >= settings.GAME_STATES_IN_DB_TO_START_TRAIN

    def add_state(self,
            state: VehicleState, next_state: VehicleState,
            reward: np.float32, ops, values, final_state: bool
        ):
        self._states.append(TrainState(state, next_state, reward, ops, values, final_state))

    def set_reward(self, reward: np.float32, t: np.float32, vehicle_id: int):
        for s in self._states:
            if (s.t == np.float32(0.0)) or (s.vehicle_id != vehicle_id):
                continue
            if t - s.t >= np.float32(0.0):
                s.reward += np.float32(math.pow(settings.RL_REWARD_COEF,
                                                (t - s.t)*settings.VEHICLE_UI_BOT_STEPS_PER_SECOND)
                                       * reward)
            s.t = np.float32(0.0)

    def _calc_avg_values(self):
        if self._states:
            self._avg_got = sum([s.getted for s in self._states])/len(self._states)
            self._avg_reward = sum([s.reward for s in self._states])/len(self._states)
            self._avg_values = sum([1 if s.has_values() else 0 for s in self._states])/len(self._states)
        else:
            self._avg_got = 0.0
            self._avg_reward = 0.0
            self._avg_values = 0.0

    @staticmethod
    def has_reward(s: TrainState, max_reward, inverse=False):
        v = s.get_reward_index(max_reward)
        if inverse:
            v = 9 - v
        return v

    def reduce_size(self):
        if len(self._states) <= self._max_length:
            return
        self._calc_avg_values()
        random.shuffle(self._states)
        self._init_max_reward()
        self._states = radix_sorted(
            self._states, 1,
            key=partial(TrainStates.has_reward, max_reward=self._max_reward, inverse=True)
        )
        length = int(self._reward_ratio*len(self._states))
        more = self._states[length:]
        self._states = self._states[:length]
        random.shuffle(more)
        self._states = self._states + more
        self._states = self._states[:self._max_length]
        self._init_indexes()
        if settings.VERBOSE_TRAIN_DB:
            print("len(Train_DB_states)={} avg_getted={:.6f} avg_reward = {:.6f} avg_values = {:0.6f}"
                .format(len(self._states), self._avg_got, self._avg_reward, self._avg_values)
                )

    def print_size(self):
        self._calc_avg_values()
        if settings.VERBOSE_TRAIN_DB:
            print("len(Train_DB_states)={} avg_getted={:.6f} avg_reward = {:.6f} avg_values = {:0.6f}"
                .format(len(self._states), self._avg_got, self._avg_reward, self._avg_values)
                )

    def _init_max_reward(self):
        rewards = [s.reward for s in self._states]
        if rewards:
            self._max_reward = max(rewards)
            self._max_reward = max([self._max_reward, settings.REINFORCE_DONE*0.2])
        else:
            self._max_reward = settings.REINFORCE_DONE * 0.2

    def _init_indexes(self):
        self._init_max_reward()
        self._indexes = np.array(
            radix_sorted_indexes(
                self._states, 1,
                key=partial(TrainStates.has_reward, max_reward=self._max_reward, inverse=True)
            ),
            dtype=np.int32)
        self._reward_count = sum(
            [1 if TrainStates.has_reward(s, max_reward = self._max_reward)>0 else 0
            for s in self._states]
        )

    def get_items(self, count: int = 1):
        if self._indexes.shape[0] != len(self._states):
            self._init_indexes()
        items_count = len(self._states)
        if self._reward_count > 0:
            p1 = max((self._reward_count/items_count, self._reward_ratio))
        else:
            p1 = self._reward_count/items_count
        p2 = (1.0 - p1)
        if items_count - self._reward_count > 0:
            p2 /= (items_count - self._reward_count)
        if self._reward_count > 0:
            p1 /= self._reward_count
        p = np.zeros((items_count,), dtype=np.float32)
        p[:self._reward_count] = np.float32(p1)
        p[self._reward_count:] = np.float32(p2)
        indexes = np.random.choice(self._indexes, count, replace=False, p=p)
        for i in range(count):
            self._states[indexes[i]].getted += 1
        return [self._states[indexes[i]] for i in range(count)]

    def get_all_items(self):
        return self._states

    def save(self, fn: AnyStr):
        with open(fn, "wb") as f:
            rp = settings.TARGET.get_reinforce_params()
            print("db save reinforce params = {}".format(rp))
            data = rp, self._states
            pickle.dump(data, f)

    def _load(self, fn: AnyStr):
        self._states = list()
        with open(fn, "rb") as f:
            reinforce_params, self._states = pickle.load(f)
            settings.TARGET.set_reinforce_params(reinforce_params)

    pass  # class TrainStates


class AsyncTrainDBProcessor:
    def __init__(self, db: TrainStates):
        self._can_process_requests = [RequestType.SAVE_TRAIN_STATE]
        self._db = db

    def process(self, request: Request):
        if not request.in_types(self._can_process_requests):
            return False
        if request.is_type(RequestType.SAVE_TRAIN_STATE):
            self._db.add_state(
                request.state, request.next_state,
                np.float32(request.reward), request.ops, request.values,
                request.final_state
            )
        return True

    pass  # class AsyncTrainDBProcessor
