from geom.utils import SingletonDecorator
from geom.primitives import angle_diff
from RLParking.settings import settings

import numpy as np
import math


def calc_fail_value(max_fail_value: np.float32, target_distance, target_max_distance):
    val = max_fail_value*(math.log(target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_fail_value):
        return max_fail_value
    else:
        return np.float32(val)


def calc_done_value(max_done_value: np.float32, target_distance, target_max_distance):
    target_distance = min((target_distance, target_max_distance))
    val = max_done_value \
          * (math.log(target_max_distance - target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_done_value):
        return max_done_value
    else:
        return np.float32(val)


def get_reward(old_values, new_values):
    assert(settings.REWARD_KEY_COLLISION in new_values)
    assert(settings.REWARD_KEY_DRIVE_TIME in new_values)
    assert((settings.REWARD_KEY_POS in old_values) and (settings.REWARD_KEY_POS in new_values))
    assert((settings.REWARD_KEY_VELOCITY in old_values) and (settings.REWARD_KEY_VELOCITY in new_values))
    assert((settings.REWARD_KEY_ANGLE in old_values) and (settings.REWARD_KEY_ANGLE in new_values))
    assert((settings.REWARD_KEY_TIME in old_values) and (settings.REWARD_KEY_TIME in new_values))

    target_dist = (new_values[settings.REWARD_KEY_POS] - settings.TARGET.TARGET_POINT).length()
    if (new_values[settings.REWARD_KEY_COLLISION]
            or (target_dist > settings.TARGET.TARGET_POINT_MAX_DISTANCE)
            or (new_values[settings.REWARD_KEY_DRIVE_TIME] > settings.VEHICLE_MAX_DRIVE_DURATION)
        ):
        return calc_fail_value(settings.REINFORCE_FAIL, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    if ((settings.REWARD_KEY_STAND_TIME in old_values)
        and (
            np.fabs(new_values[settings.REWARD_KEY_TIME] - old_values[settings.REWARD_KEY_STAND_TIME])
            > settings.VEHICLE_MAX_STAND_DURATION)
        ):
        return calc_fail_value(settings.REINFORCE_FAIL, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    angle_dist = angle_diff(new_values[settings.REWARD_KEY_ANGLE], settings.TARGET.TARGET_ANGLE)

    # bound to [0 .. math.pi*0.5]
    angle_dist = np.float32(math.fmod(angle_dist + math.pi, math.pi))
    angle_dist = np.float32(math.fmod(angle_dist + math.pi, math.pi))
    if angle_dist > math.pi*0.5:
        angle_dist = np.float32(math.pi - angle_dist)

    velocity_dist = new_values[settings.REWARD_KEY_VELOCITY]
    td = settings.TARGET.TARGET_POINT_REINFORCE_DISTANCE * settings.TARGET.TARGET_REINFORCE_BORDER
    ta = settings.TARGET.TARGET_ANGLE_REINFORCE_DISTANCE * settings.TARGET.TARGET_REINFORCE_BORDER
    tv = settings.TARGET.TARGET_VELOCITY_REINFORCE_DISTANCE*settings.TARGET.TARGET_REINFORCE_BORDER
    if ((target_dist <= td)
        and (angle_dist <= ta)
        and (np.fabs(velocity_dist) <= tv)
        ):
        settings.TARGET.update_reinforce_distances(target_dist, angle_dist, velocity_dist)
        return calc_done_value(settings.REINFORCE_DONE, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    return settings.REINFORCE_NONE, False


class Reward:
    def __init__(self):
        self._old_values = dict()
        self._last_id = 0

    def set_values(self, obj_id, values_dict):
        if obj_id not in self._old_values:
            self._old_values[obj_id] = dict()
        for k, v in values_dict.items():
            self._old_values[obj_id][k] = v

    def get_reward(self, obj_id, new_values_dict):
        if obj_id not in self._old_values:
            return settings.REINFORCE_NONE
        stand_pos = None
        stand_time = None
        if ((settings.REWARD_KEY_STAND_POS in self._old_values[obj_id])
                and (settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]
            if not np.isclose((stand_pos - new_values_dict[settings.REWARD_KEY_POS]).length(), np.float32(0.0)):
                stand_pos = None
                stand_time = None

        if stand_pos is not None:
            self._old_values[obj_id][settings.REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME] = stand_time
        else:
            if settings.REWARD_KEY_STAND_POS in self._old_values[obj_id]:
                del self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            if settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id]:
                del self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]

        val = get_reward(self._old_values[obj_id] if obj_id in self._old_values else dict(), new_values_dict)

        if ((settings.REWARD_KEY_STAND_POS in self._old_values[obj_id])
            and (settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]
        elif np.isclose(
                (self._old_values[obj_id][settings.REWARD_KEY_POS]
                 - new_values_dict[settings.REWARD_KEY_POS]).length()
                , np.float32(0.0)
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_TIME]
        else:
            stand_pos = None
            stand_time = None

        del self._old_values[obj_id]

        if stand_pos is not None:
            self._old_values[obj_id] = dict()
            self._old_values[obj_id][settings.REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME] = stand_time
        return val

    def get_next_id(self):
        res = self._last_id
        self._last_id += 1
        return res

    pass  # class Reward


Reward = SingletonDecorator(Reward)
