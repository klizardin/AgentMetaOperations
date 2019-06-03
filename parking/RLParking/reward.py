import numpy as np
from geom.utils import SingletonDecorator
from geom.primitives import angle_diff
from RLParking.settings import *


def calc_fail_value(max_fail_value: np.float32, target_distance, target_max_distance):
    val = max_fail_value*(math.log(target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_fail_value):
        return max_fail_value
    else:
        return np.float32(val)


def calc_done_value(max_done_value: np.float32, target_distance, target_max_distance):
    target_distance = min((target_distance,target_max_distance))
    val = max_done_value \
          * (math.log(target_max_distance - target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_done_value):
        return max_done_value
    else:
        return np.float32(val)


def get_reward(old_values, new_values):
    assert(REWARD_KEY_COLLISION in new_values)
    assert(REWARD_KEY_DRIVE_TIME in new_values)
    assert((REWARD_KEY_POS in old_values) and (REWARD_KEY_POS in new_values))
    assert((REWARD_KEY_VELOCITY in old_values) and (REWARD_KEY_VELOCITY in new_values))
    assert((REWARD_KEY_ANGLE in old_values) and (REWARD_KEY_ANGLE in new_values))
    assert((REWARD_KEY_TIME in old_values) and (REWARD_KEY_TIME in new_values))

    target_dist = (new_values[REWARD_KEY_POS] - TARGET_POINT).length()
    if (new_values[REWARD_KEY_COLLISION]
            or (target_dist > TARGET_POINT_MAX_DISTANCE)
            or (new_values[REWARD_KEY_DRIVE_TIME] > VEHICLE_MAX_DRIVE_DURATION)
        ):
        return calc_fail_value(REINFORCE_FAIL, target_dist, TARGET_POINT_MAX_DISTANCE), True

    if ((REWARD_KEY_STAND_TIME in old_values)
        and (
            np.fabs(new_values[REWARD_KEY_TIME] - old_values[REWARD_KEY_STAND_TIME])
            > VEHICLE_MAX_STAND_DURATION)
        ):
        return calc_fail_value(REINFORCE_FAIL, target_dist, TARGET_POINT_MAX_DISTANCE), True

    angle_dist = angle_diff(new_values[REWARD_KEY_ANGLE],TARGET_ANGLE)

    # bound to [0 .. math.pi*0.5]
    angle_dist = np.float32(math.fmod(angle_dist, math.pi))
    if angle_dist < 0:
        angle_dist = np.float32(angle_dist + math.pi)
    if angle_dist > math.pi*0.5:
        angle_dist = np.float32(math.pi - angle_dist)

    velocity_dist = new_values[REWARD_KEY_VELOCITY]
    if (np.isclose(
            target_dist, np.float32(0.0),
            atol=TARGET_POINT_REINFORCE_DISTANCE*TARGET_REINFORCE_BORDER)
        and np.isclose(
                angle_dist, np.float32(0.0),
                atol=TARGET_ANGLE_REINFORCE_DISTANCE*TARGET_REINFORCE_BORDER)
        and np.isclose(
                velocity_dist, np.float32(0.0),
                atol=TARGET_VELOCITY_REINFORCE_DISTANCE*TARGET_REINFORCE_BORDER)
        ):
        update_reinforce_distances(target_dist, angle_dist, velocity_dist)
        return calc_done_value(REINFORCE_DONE, target_dist, TARGET_POINT_MAX_DISTANCE), True

    return (REINFORCE_NONE, False)


class Reward:
    def __init__(self):
        self._old_values = dict()
        self._last_id = 0

    def set_values(self, obj_id, values_dict):
        if obj_id not in self._old_values:
            self._old_values[obj_id] = dict()
        for k,v in values_dict.items():
            self._old_values[obj_id][k] = v

    def get_reward(self, obj_id, new_values_dict):
        if obj_id not in self._old_values:
            return REINFORCE_NONE
        stand_pos = None
        stand_time = None
        if ((REWARD_KEY_STAND_POS in self._old_values[obj_id])
                and (REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][REWARD_KEY_STAND_TIME]
            if not np.isclose(
                (stand_pos - new_values_dict[REWARD_KEY_POS]).length()
                , np.float32(0.0)
                ):
                stand_pos = None
                stand_time = None

        if stand_pos is not None:
            self._old_values[obj_id][REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][REWARD_KEY_STAND_TIME] = stand_time
        else:
            if REWARD_KEY_STAND_POS in self._old_values[obj_id]:
                del self._old_values[obj_id][REWARD_KEY_STAND_POS]
            if REWARD_KEY_STAND_TIME in self._old_values[obj_id]:
                del self._old_values[obj_id][REWARD_KEY_STAND_TIME]

        val = get_reward(self._old_values[obj_id] if obj_id in self._old_values else dict(), new_values_dict)

        if ((REWARD_KEY_STAND_POS in self._old_values[obj_id])
            and (REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][REWARD_KEY_STAND_TIME]
        elif np.isclose(
                (self._old_values[obj_id][REWARD_KEY_POS] - new_values_dict[REWARD_KEY_POS]).length()
                , np.float32(0.0)
            ):
            stand_pos = self._old_values[obj_id][REWARD_KEY_POS]
            stand_time = self._old_values[obj_id][REWARD_KEY_TIME]
        else:
            stand_pos = None
            stand_time = None

        del self._old_values[obj_id]

        if stand_pos is not None:
            self._old_values[obj_id] = dict()
            self._old_values[obj_id][REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][REWARD_KEY_STAND_TIME] = stand_time
        return val

    def get_next_id(self):
        res = self._last_id
        self._last_id += 1
        return res

    pass # class Reward


Reward = SingletonDecorator(Reward)