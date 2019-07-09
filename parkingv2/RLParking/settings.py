from typing import AnyStr
import numpy as np
import math
import tensorflow as tf
from geom.primitives import Point, Size, Line, Rect, angle_diff
import os


def kminh_to_mins(val):
    return np.float32(val)*np.float32(1000.0/(60*60))


def acceleration(val, t, c = 1.0):
    return np.float32(val)/np.float32(t)*np.float32(c)


def grad_to_rad(a):
    return np.float32(a)*np.float32(math.pi/180)


def rotate_acceleration(a, t, c = 1.0):
    return grad_to_rad(a)/np.float32(t)*np.float32(c)


class VehicleConsts:
    TRANSMISSION_ACCELERATION = np.array([
        -acceleration(kminh_to_mins(40), 40, 0.6),
        -acceleration(kminh_to_mins(40), 10, 0.1),
        np.float32(0.0),
        acceleration(kminh_to_mins(40), 10, 0.1),
        acceleration(kminh_to_mins(40), 10, 0.6),
        ], dtype=np.float32)

    TRANSMISSION_SPEED = np.array([
        kminh_to_mins(-7.5),
        kminh_to_mins(-5.0),
        kminh_to_mins(0),
        kminh_to_mins(5.0),
        kminh_to_mins(10.0),
        kminh_to_mins(15.0),
        ], dtype=np.float32)

    SMALL_SPEED_COEF = np.float32(1e-4)

    ROTATE_ACCELERATION = np.array([
        -rotate_acceleration(45.0, 1.0, 1.0),
        -rotate_acceleration(45.0, 1.0, 0.1),
        np.float32(0.0),
        rotate_acceleration(45.0, 1.0, 0.1),
        rotate_acceleration(45.0, 1.0, 1.0),
        ], dtype=np.float32)

    ROTATE_ANGLE_BOUNDS = np.array([
        grad_to_rad(-45.0),
        grad_to_rad(45.0)
    ], dtype=np.float32)

    BOUNDS_RECT = [Point(2, 1), Point(2, -1), Point(-2, -1), Point(-2, 1)] # points of the
    CENTER_PT = Point(0, 0)
    AXLE_FRONT = [Point(1.5, 1), Point(1.5, -1)]
    AXLE_BACK = [Point(-1.5, 1), Point(-1.5, -1)]
    WHEEL_SIZE = np.float32(0.5)

    VERBOSE = True

    pass #class VehicleConsts

VEHICLE = VehicleConsts()

SCREEN_SIZE = Size(cx=50, cy=50)
STATIC_VEHICLES = [
    (VEHICLE.BOUNDS_RECT, VEHICLE.CENTER_PT, Point(6.5, 5.5), np.float32(math.pi*0.5)),
    (VEHICLE.BOUNDS_RECT, VEHICLE.CENTER_PT, Point(6.5, -5.5), np.float32(math.pi*0.5)),
]

WALLS = [
    Line(Point(8, 15), Point(8, -15)),
    Line(Point(8, 15), Point(-3, 15)),
    Line(Point(-3, 15), Point(-3, -15)),
    Line(Point(-3, -15), Point(8, -15)),
]

class Target:
    """class to update correctly"""

    def __init__(self):
        self.TARGET_POINT = Point(6.5, 0)
        self.TARGET_ANGLE = np.float32(math.pi*0.5)
        self.TARGET_POSITION_DEVIATION = np.float32(5.0) #6.5
        self.TARGET_ANGLE_DEVIATION = np.float32(math.pi*0.2) #0.2
        self.TARGET_POSITION_RB = np.float32(3.0) # 2.0

        self.TARGET_POINT_REINFORCE_DISTANCE = np.float32(8.0)
        self.TARGET_POINT_REINFORCE_DISTANCE_NEAR = np.float32(0.5)
        self.TARGET_ANGLE_REINFORCE_DISTANCE = np.float32(math.pi*0.5)
        self.TARGET_VELOCITY_REINFORCE_DISTANCE = np.float32(5.0)
        self.TARGET_REINFORCE_BORDER = np.float32(0.9)

        self.TARGET_POINT_MAX_DISTANCE = np.float32(17.0)
        #self.TARGET_CHANGE_LAMBDA = np.float32(3.33333333e-5)
        self.TARGET_CHANGE_LAMBDA = np.float32(3.333333333e-6)

    def update_reinforce_distances(self, target_dist, angle_dist, velocity_dist):
        target_dist = min((self.TARGET_POINT_REINFORCE_DISTANCE, target_dist))
        l = self.TARGET_CHANGE_LAMBDA
        if self.TARGET_POINT_REINFORCE_DISTANCE <= self.TARGET_POINT_REINFORCE_DISTANCE_NEAR:
            l = math.pow(l, 2.0)
        self.TARGET_POINT_REINFORCE_DISTANCE += np.float32(target_dist - self.TARGET_POINT_REINFORCE_DISTANCE)*l
        if self.TARGET_POINT_REINFORCE_DISTANCE <= self.TARGET_POINT_REINFORCE_DISTANCE_NEAR:
            angle_dist = min((self.TARGET_ANGLE_REINFORCE_DISTANCE, angle_dist))
            self.TARGET_ANGLE_REINFORCE_DISTANCE += np.float32(angle_dist - self.TARGET_ANGLE_REINFORCE_DISTANCE)*l
            velocity_dist = min((self.TARGET_VELOCITY_REINFORCE_DISTANCE, math.fabs(velocity_dist)))
            self.TARGET_VELOCITY_REINFORCE_DISTANCE += np.float32(velocity_dist - self.TARGET_VELOCITY_REINFORCE_DISTANCE)*l

    def get_reinforce_params(self):
        return self.TARGET_POINT_REINFORCE_DISTANCE, self.TARGET_ANGLE_REINFORCE_DISTANCE, self.TARGET_VELOCITY_REINFORCE_DISTANCE

    def set_reinforce_params(self, params):
        self.TARGET_POINT_REINFORCE_DISTANCE, self.TARGET_ANGLE_REINFORCE_DISTANCE, self.TARGET_VELOCITY_REINFORCE_DISTANCE = params
        print("set reinforce params = ({}, {}, {})"
              .format(self.TARGET_POINT_REINFORCE_DISTANCE,
                      self.TARGET_ANGLE_REINFORCE_DISTANCE,
                      self.TARGET_VELOCITY_REINFORCE_DISTANCE
                      )
              )

    pass  # Target

TARGET = Target()  # need to correctly update params


VISUALIZE_START_POSITIONS_WEIGHTS = False
BASE_POS1_FOR_POS_GENERATOR = Point(-10.0, 15.0)
BASE_POS2_FOR_POS_GENERATOR = Point(-10.0, -15.0)
MAX_LENGTH_FOR_POS_GENERATOR = 20
VERSION = "0.0.1"

MAX_INPUT_LINE_LENGTH = np.float32(15.0)
MIN_INPUT_LINE_LENGTH = np.float32(0.999)

INPUT_SHOW_RECT = Rect(top_left = Point(-20.0, -24.0), bottom_right=Point(20.0, -19.0))
NET_SHOW_RECT = Rect(top_left = Point(-20.0, 19.0), bottom_right=Point(20.0, 24.0))

VISUALIZE_INPUT_HISTORY = True
VEHICLE_STATE_MAX_LENGTH = 16
VEHICLE_STATE_INDEXES = [0,3,15]

START_POSITIONS = [
    Line(Point(4.5, 12), Point(-1.5, 12)),
    Line(Point(4.5, -12), Point(-1.5, -12)),
]


class PathesFunc:
    def __init__(self, name, func):
        self._name = name
        self._func = func
        pass

    def f(self):
        return self._func(self._name)

    pass  # class Func

class Pathes:
    def __init__(self, version):
        base_path = "/home/klizardin/tmp/parking/{0}/".format(version)
        self._pathes = {
            "tmp": base_path,
            "base_coordinates": os.path.join(base_path, "base_coordinates/") + "coords",
            "train_state": os.path.join(base_path, "train/") + "states",
            "main_model": os.path.join(base_path, "train/") + "main_model.h5",
        }
        #declare functions
        self.get_temp = PathesFunc("tmp", self._get_path).f
        self.is_temp_exist = PathesFunc("tmp", self._is_exist).f
        self.get_base_coordinates = PathesFunc("base_coordinates", self._get_path).f
        self.has_base_coordinates = PathesFunc("base_coordinates", self._is_exist).f
        self.get_train_state = PathesFunc("train_state", self._get_path).f
        self.has_train_state = PathesFunc("train_state", self._is_exist).f
        self.get_main_model = PathesFunc("main_model", self._get_path).f
        self.has_main_model = PathesFunc("main_model", self._is_exist).f

    def _create_path(self, path: AnyStr):
        p,f = os.path.split(path)
        if not os.path.isdir(p):
            os.makedirs(p)

    def _get_path(self, name):
        assert(name in self._pathes)
        path = self._pathes[name]
        self._create_path(path)
        return path

    def _is_exist(self, name):
        assert (name in self._pathes)
        path = self._pathes[name]
        return os.path.isdir(path) or os.path.isfile(path)

    pass #class Pathes


PATHES = Pathes(VERSION)
BASE_COORDINATES_COUNT = 10000

SENSOR_INPUTS_SIZE = 4 + len(VEHICLE_STATE_INDEXES)*3

ANGLE_FOR_INPUT = 30
ANGLES_FOR_OPERATIONS = 20
ANGLE_FOR_OPERATION_MIN = grad_to_rad(-30.0)
ANGLE_FOR_OPERATION_MAX = grad_to_rad(30.0)
OPERATIONS_COUNT = ANGLES_FOR_OPERATIONS + 2
NET_INPUT_SIZE = int((360//ANGLE_FOR_INPUT)*len(VEHICLE_STATE_INDEXES) + SENSOR_INPUTS_SIZE + OPERATIONS_COUNT)
NET1_FC_SIZE1 = 512
NET_OPERATION_EXTRA_MIN_SIZE = 128
NET_OPERATION_ITEM_SIZE = int((NET1_FC_SIZE1 - NET_OPERATION_EXTRA_MIN_SIZE)//OPERATIONS_COUNT)
NET_OPERATION_EXTRA_SIZE = NET1_FC_SIZE1 - NET_OPERATION_ITEM_SIZE*OPERATIONS_COUNT
NET1_FC_SIZE2 = 512
NET1_FC_SIZE3 = 512
NET_LAYER1_ACTIVATION = tf.nn.leaky_relu
NET_LAYER2_ACTIVATION = tf.nn.leaky_relu
NET1_FC_DROPOUT_VALUE1 = 0.2
NET1_FC_DROPOUT_VALUE2 = 0.2


VERBOSE_TRAIN_DB = True
VERBOSE_VEHICLE_COMMANDS = False
VERBOSE_MAIN_MODEL_SAVE = True

REWARD_KEY_COLLISION = "collision"
REWARD_KEY_POS = "pos"
REWARD_KEY_STAND_POS = "stand_pos"
REWARD_KEY_STAND_TIME = "stand_time"
REWARD_KEY_VELOCITY = "velocity"
REWARD_KEY_ANGLE = "angle"
REWARD_KEY_TIME = "time"
REWARD_KEY_DRIVE_TIME = "drive_time"

REINFORCE_DONE = np.float32(1.0)
REINFORCE_FAIL = np.float32(-1.0)
REINFORCE_NONE = np.float32(0.0)

REWARD_TO_VALUES_COEF = np.float32(0.3333333)
VALUES_TO_REWARD_COEF = np.float32(1.0/REWARD_TO_VALUES_COEF)
RL_RO_COEF = np.float32(0.99)
RL_REWARD_COEF = np.float32(0.99)

STATES_TO_TRAIN_BATCH_SIZE=32
STATES_TO_TRAIN=STATES_TO_TRAIN_BATCH_SIZE*4

MAIN_MODEL_SAVE_INTERVAL = 60*25
VALUE_MORE_THAN_MAX_NET_RESULT = 1e2

RL_SEARCH_COEF = 0.01
RL_PRETRAIN_SEARCH_COEF = 0.1
RL_SEARCH_USE_ALPHA_ALGORITHM = False
USE_LEARNING_BOOST = True

VEHICLE_UI_BOT_STEPS_PER_SECOND = np.float32(3.0)
VEHICLE_STEP_DURATION = np.float32(1.0/VEHICLE_UI_BOT_STEPS_PER_SECOND)
VEHICLE_MAX_DRIVE_DURATION = np.float32(60.0*2.0)
VEHICLE_MAX_STAND_DURATION = np.float32(5)

GAME_STATES_IN_DB_SIZE_MAX = 1024*31
GAME_STATES_IN_DB_SIZE_MAX2 = 1024*32
GAME_STATES_IN_DB_TO_START_TRAIN = 1024*30
GAME_STATES_IN_DB_STEP_SIZE = 100
GAME_STATES_IN_DB_REWARD_RATIO = np.float32(0.3333333)
GAME_STATE_GETTED_MAX_COUNT = 30

STATE_LEARN_VALUES_COUNT = 8
SET_OPERATION_VALUES_PROB = np.float32(0.1)


class ResetWeightCoefs:
    RESET_WEIGHTS_COEF = np.float32(1.0)

    def clear(self):
        self.RESET_WEIGHTS_COEF = np.float32(1.0)

    def set(self):
        self.RESET_WEIGHTS_COEF = np.float32(5e-1)

    pass  # class ResetWeightCoefs


RESET_WEIGHTS_COEF = ResetWeightCoefs()

