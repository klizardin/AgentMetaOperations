from RLParking.settings import settings
from geom.primitives import Point, Line
from geom.utils import froze_class
from RLParking.reward import Reward
from geom.primitives import angle_diff

from enum import Enum
import copy
from typing import List, AnyStr
import scipy
import numpy as np
import math


def get_lines(bounds):
    pt0 = bounds[-1]
    lines = list()
    for pt in bounds:
        lines.append(Line(pt0, pt))
        pt0 = pt
    return lines


class StaticVehicle:
    def __init__(self, static_vehicle_pos_info):
        bp = static_vehicle_pos_info[1]
        self._pos = static_vehicle_pos_info[2]
        self._angle = static_vehicle_pos_info[3]
        self._pts = [
            pt.rotate(bp, self._angle) + self._pos
            for pt in static_vehicle_pos_info[0]
            ]

    @property
    def bounds(self):
        return [copy.copy(pt) for pt in self._pts]

    @property
    def position(self):
        return copy.copy(self._pos)

    @property
    def rotation(self):
        return self._angle

    pass  # class StaticVehicle


class StaticEnv:
    def __init__(self):
        self.static_vehicles = [StaticVehicle(svi) for svi in settings.STATIC_VEHICLES]
        self.walls = settings.WALLS
        pass

    def get_lines(self):
        all_lines = list()
        for sv in self.static_vehicles:
            lines = get_lines(sv.bounds)
            all_lines.extend(lines)
        all_lines.extend(self.walls)
        return all_lines

    pass  # class StaticEnv


class VehicleOp(Enum):
    NONE = 0
    ACCELERATE_BACKWARD_2 = 1
    ACCELERATE_BACKWARD_1 = 2
    ACCELERATE_NONE = 3
    ACCELERATE_FORWARD_1 = 4
    ACCELERATE_FORWARD_2 = 5
    ROTATE_RIGHT_2 = 7
    ROTATE_RIGHT_1 = 7
    ROTATE_NONE = 8
    ROTATE_LEFT_1 = 9
    ROTATE_LEFT_2 = 10

    pass  # class VehicleOp


VEHICLE_ACCELERATE_OPS = [
    VehicleOp.ACCELERATE_BACKWARD_2, VehicleOp.ACCELERATE_BACKWARD_1,
    VehicleOp.ACCELERATE_NONE,
    VehicleOp.ACCELERATE_FORWARD_1, VehicleOp.ACCELERATE_FORWARD_2
]


VEHICLE_ROTATE_OPS = [
    VehicleOp.ROTATE_RIGHT_2, VehicleOp.ROTATE_RIGHT_1,
    VehicleOp.ROTATE_NONE,
    VehicleOp.ROTATE_LEFT_1, VehicleOp.ROTATE_LEFT_2
]


# VEHICLE_TRANSMITE_OPS = [VehicleOp.FORWARD, VehicleOp.BACKWARD]


VEHICLE_POSSIBLE_OPERATIONS = [
    (VehicleOp.NONE,),
]

# for o in VEHICLE_TRANSMITE_OPS:
#    VEHICLE_POSSIBLE_OPERATIONS.append((o,))


def init_all_ops(pos_ops):
    for rotate_op in VEHICLE_ROTATE_OPS:
        for angle_op in VEHICLE_ACCELERATE_OPS:
            if ((angle_op.value == VehicleOp.ACCELERATE_NONE.value)
                and (rotate_op.value == VehicleOp.ROTATE_NONE.value)
            ):
                continue
            pos_ops.append((rotate_op, angle_op))
    return pos_ops


VEHICLE_POSSIBLE_OPERATIONS = init_all_ops(VEHICLE_POSSIBLE_OPERATIONS)


def get_breaks_count():
    return int(np.sum(np.where(settings.VEHICLE.TRANSMISSION_ACCELERATION < 0.0, 1, 0)))


def get_accelaration_count():
    return int(np.sum(np.where(settings.VEHICLE.TRANSMISSION_ACCELERATION > 0.0, 1, 0)))


@froze_class
class VehicleOperation:
    def __init__(self, op_id: VehicleOp, name: AnyStr, forward: bool = None,
                 acceleration: int = None, rotate: int = None):
        self.op_id = op_id
        self.name = name
        self.forward = forward
        if acceleration is not None:
            self.break_value = get_breaks_count() - acceleration if acceleration <= get_breaks_count() else None
            self.accelaration_value = acceleration - get_breaks_count() if acceleration > get_breaks_count() else None
        else:
            self.break_value = None
            self.accelaration_value = None
        self.rotate = rotate

    pass  # class VehicleOperation


VEHICLE_SINGLE_OPERATIONS = {
    VehicleOp.NONE:  VehicleOperation(VehicleOp.NONE, "none"),
    # VehicleOp.FORWARD : VehicleOperation(VehicleOp.FORWARD, "forward", forward=True),
    # VehicleOp.BACKWARD : VehicleOperation(VehicleOp.BACKWARD, "backward", forward=False),
    VehicleOp.ACCELERATE_BACKWARD_2: VehicleOperation(VehicleOp.ACCELERATE_BACKWARD_2, "break_2", acceleration=0),
    VehicleOp.ACCELERATE_BACKWARD_1: VehicleOperation(VehicleOp.ACCELERATE_BACKWARD_1, "break_1", acceleration=1),
    VehicleOp.ACCELERATE_NONE: VehicleOperation(VehicleOp.ACCELERATE_NONE, "accelerate_0", acceleration=2),
    VehicleOp.ACCELERATE_FORWARD_1: VehicleOperation(VehicleOp.ACCELERATE_FORWARD_1, "accelerate_1", acceleration=3),
    VehicleOp.ACCELERATE_FORWARD_2: VehicleOperation(VehicleOp.ACCELERATE_FORWARD_2, "accelerate_2", acceleration=4),
    VehicleOp.ROTATE_LEFT_2: VehicleOperation(VehicleOp.ROTATE_LEFT_2, "left_2", rotate=-2),
    VehicleOp.ROTATE_LEFT_1: VehicleOperation(VehicleOp.ROTATE_LEFT_1, "left_1", rotate=-1),
    VehicleOp.ROTATE_NONE: VehicleOperation(VehicleOp.ROTATE_NONE, "rotate_none", rotate=0),
    VehicleOp.ROTATE_RIGHT_1: VehicleOperation(VehicleOp.ROTATE_RIGHT_1, "right_1", rotate=1),
    VehicleOp.ROTATE_RIGHT_2: VehicleOperation(VehicleOp.ROTATE_RIGHT_2, "right_2", rotate=2),
}

# VEHICLE_POSSIBLE_OPERATIONS = [
#    (VEHICLE_SINGLE_OPERATIONS[i] for i in line)
#    for line in VEHICLE_POSSIBLE_OPERATIONS
# ]


@froze_class
class VehiclePos:
    def __init__(self, pos: Point, angle: np.float32, t: np.float32):
        self.pos = pos
        self.angle = angle
        self.t = t

    pass  # class VehiclePos


class VehicleState:
    def __init__(self):
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)
        self.front = True
        self.wheel_angle = np.float32(0.0)
        self.wheel_delta_angle = np.float32(0.0)
        self.states = list()
        self.commads = (VehicleOp.NONE,)
        self.last_transmite_op = None
        self.last_rotate_op = None
        self.last_accelerate_op = None
        self.clear()

    def add(self, dt: np.float32):
        p = VehiclePos(copy.deepcopy(self.pos), self.angle, self.last_time + dt)
        if len(self.states) < settings.VEHICLE_STATE_MAX_LENGTH:
            self.states.append(p)
        else:
            self.states[:-1] = self.states[1:]
            self.states[-1] = p

    def replace(self, pos: Point, angle: np.float32, t: np.float32):
        p = VehiclePos(pos, angle, t)
        self.states[-1] = p

    def get_at(self, i):
        i = min((i+1, len(self.states)))
        return self.states[-i]

    @property
    def pos(self):
        return self.states[-1].pos

    @pos.setter
    def pos(self, p: Point):
        self.states[-1].pos = p

    @property
    def angle(self):
        return self.states[-1].angle

    @angle.setter
    def angle(self, angle):
        self.states[-1].angle = np.float32(angle)

    @property
    def last_time(self):
        return self.states[-1].t

    def set_current(self,
            velocity: np.float32, acceleration: np.float32,
            front: bool,
            wheel_angle: np.float32, wheel_delta_angle: np.float32):
        self.velocity = velocity
        self.acceleration = acceleration
        self.front = front
        self.wheel_angle = wheel_angle
        self.wheel_delta_angle = wheel_delta_angle

    def clear(self):
        self.states = list()
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)
        self.front = True
        self.wheel_angle = np.float32(0.0)
        self.wheel_delta_angle = np.float32(0.0)
        self.states = list()
        self.states.append(VehiclePos(Point(0, 0), np.float32(0.0), np.float32(0.0)))
        self.commads = (VehicleOp.NONE,)
        self.last_transmite_op = None
        self.last_rotate_op = None
        self.last_accelerate_op = None

    @staticmethod
    def _cmds_of_ops(cmds, ops):
        return [
            op
            for op in cmds
            if any([top.value == op.value for top in ops])
        ]

    def set_commands(self, cmds):
        self.commads = cmds
        # transmite_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_TRANSMITE_OPS)
        # if transmite_cmds:
        #    self.last_transmite_op = transmite_cmds[0]
        rotate_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_ROTATE_OPS)
        if rotate_cmds:
            self.last_rotate_op = rotate_cmds[0]
        accelerate_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_ACCELERATE_OPS)
        if accelerate_cmds:
            self.last_accelerate_op = accelerate_cmds[0]

    def get_last_commands(self):
        return self.commads

    def stop(self):
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)

    def delta_time(self):
        return self.states[-1].t - self.states[0].t

    def to_reward_values(self):
        return {
            settings.REWARD_KEY_POS: self.pos,
            settings.REWARD_KEY_ANGLE: self.angle,
            settings.REWARD_KEY_VELOCITY: self.velocity,
            settings.REWARD_KEY_TIME: self.last_time
        }

    pass  # class VehicleState


class Vehicle:
    def __init__(self, name):
        self._name = name
        self._state = VehicleState()
        self._reward = Reward()
        self._vehicle_id = self._reward.get_next_id()
        self._drive_time = np.float32(0.0)

    @property
    def name(self):
        return self._name

    @property
    def vehicle_id(self):
        return self._vehicle_id

    @property
    def front(self):
        return self._state.front

    @property
    def acceleration(self):
        return self._state.acceleration

    @property
    def velocity(self):
        return self._state.velocity

    def _set_small_velocity(self):
        self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[-1]*settings.VEHICLE.SMALL_SPEED_COEF

    @property
    def rotation(self):
        return self._state.angle

    @property
    def wheel_angle(self):
        return self._state.wheel_angle

    @property
    def wheel_delta_angle(self):
        return self._state.wheel_delta_angle

    @property
    def position(self):
        return copy.deepcopy(self._state.pos)

    @property
    def state(self):
        return copy.deepcopy(self._state)

    @state.setter
    def state(self, s):
        self._state = copy.deepcopy(s)

    @property
    def saved_interval(self):
        return self._state.delta_time()

    def reset(self):
        self._state.stop()

    def initialize(self, pos: Point, angle, velocity, lines: List[Line] = None):
        self.reset()
        self._state.clear()
        if (lines is not None) and self._check_collision(pos, np.float32(angle), lines):
            return False
        self._state.replace(pos, angle, np.float32(0.0))
        self._state.velocity = np.float32(velocity)
        self._drive_time = np.float32(0.0)
        return True

    def setup_reward(self):
        self._reward.set_values(self._vehicle_id, self._state.to_reward_values())

    def commands(self, cmds: List[VehicleOp]):
        if not cmds:
            return
        self.setup_reward()
        if settings.VERBOSE_VEHICLE_COMMANDS:
            print("Commands:")
        for cmd_key in cmds:
            cmd = VEHICLE_SINGLE_OPERATIONS[cmd_key]
            if settings.VERBOSE_VEHICLE_COMMANDS:
                print("\t{}".format(cmd.name))
            if cmd.forward is not None:
                self.command_transmission(cmd.forward)
                continue
            if cmd.break_value is not None:
                self.command_backward(cmd.break_value)
            elif cmd.accelaration_value is not None:
                self.command_accelerate(cmd.accelaration_value)
            if cmd.rotate is not None:
                self.command_turn(cmd.rotate)
        self._state.set_commands(cmds)

    # def _can_use_transmite_commands(self):
    #    return np.isclose(self._state.velocity, np.float32(0.0), atol=1e-1)

    def get_next_available_ops(self, filter_ops: bool = False):
        cmds = [(VehicleOp.NONE,)]
        # if self._can_use_transmite_commands():
        #    cmds.extend([(o,) for o in VEHICLE_TRANSMITE_OPS])
        for rot_op in VEHICLE_ROTATE_OPS:
            for angle_op in VEHICLE_ACCELERATE_OPS:
                # if filter_ops:
                #    if (self._state.last_rotate_op is not None
                #        and abs(self._state.last_rotate_op.value - rot_op.value) > 1
                #        ):
                #        continue
                #    if (self._state.last_accelerate_op is not None
                #        and abs(self._state.last_accelerate_op.value - angle_op.value) > 1
                #        ):
                #        continue
                if (
                    (rot_op.value == VehicleOp.ROTATE_NONE.value)
                    and (angle_op.value == VehicleOp.ACCELERATE_NONE.value)
                    ):
                    continue
                # if (np.isclose(self._state.velocity, np.float32(0.0))
                #    and (angle_op.value < VehicleOp.ACCELERATE_NONE.value)
                #    ):
                #    continue
                cmds.append((rot_op, angle_op))
        return cmds

    def command_transmission(self, front: bool):
        """
        set up transmission of vehicle
        :param front: True or False - move forward or backward
        :return: True if command have been done ok
        """
    #    if self._can_use_transmite_commands():
    #        self._state.front = front
    #        return True
    #    elif VEHICLE.VERBOSE:
    #        print("{} is not stoped to switch transmission to {}"
    #              .format(self._name, "forward" if front else "backward"))
    #        return False
        pass

    def get_backward_count(self):
        """
        :return: break commands count [0, get_break_count()]
        """
        return get_breaks_count()

    def get_acceleration_count(self):
        """
        :return: acceleration commands count [0, get_acceleration_count()]
        """
        return get_accelaration_count()

    def command_backward(self, value: int):
        """
        implement break command
        :param value: [0, get_break_count()] possible commands to lower speed of vehicle
        :return: True if command have been done ok
        """
        bc = self.get_backward_count()
        if (value < 0) or (value > bc):
            return False
        self._state.acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[bc - value]
        return True

    def command_accelerate(self, value: int):
        """
        :param value: [0, get_acceleration_count()] possible commands to accelerate vihicle
        :return: True if command done have been ok
        """
        bc = self.get_backward_count()
        ac = self.get_acceleration_count()
        if (value < 0) or (value > ac):
            return False
        self._state.acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[value + bc]
        return True

    def get_rotate_count(self):
        """
        :return: rotate commands count [-get_rotate_count(), get_rotate_count()]
        """
        return settings.VEHICLE.ROTATE_ACCELERATION.shape[0] // 2

    def command_turn(self, value: int):
        """
        :param value: [-get_rotate_count(), get_rotate_count()] command to turn
        :return: True if command have been done ok
        """
        rc = self.get_rotate_count()
        if (value < -rc) or (value > rc):
            return False
        self._state.wheel_delta_angle = settings.VEHICLE.ROTATE_ACCELERATION[rc + value]
        return True

    def _velocity_step(self, t: np.float32):
        self._state.velocity += self._state.acceleration * t  # * np.float32(1.0 if self.front else -1.0)
        assert(settings.VEHICLE.TRANSMISSION_SPEED[-1] > 0)
        if self._state.velocity > settings.VEHICLE.TRANSMISSION_SPEED[-1]:
            self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[-1]
        assert(settings.VEHICLE.TRANSMISSION_SPEED[0] < 0)
        if self._state.velocity < settings.VEHICLE.TRANSMISSION_SPEED[0]:
            self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[0]
        return True

    def _turn_step(self, t: np.float32):
        self._state.wheel_angle += self._state.wheel_delta_angle * t
        assert((settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0] < 0) and (settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1] > 0))
        if self._state.wheel_angle < settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0]:
            self._state.wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0]
        if self._state.wheel_angle > settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1]:
            self._state.wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1]
        return True

    def _get_bounds(self, pos: Point, angle: np.float32):
        return [
            pt.rotate(settings.VEHICLE.CENTER_PT, angle) + pos
            for pt in settings.VEHICLE.BOUNDS_RECT
        ]

    def get_bounds(self):
        return self._get_bounds(self._state.pos, self._state.angle)

    def get_axles(self):
        pts = [
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_FRONT
        ]
        pts.extend([
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_BACK
        ])
        return pts

    def get_front_wheels(self):
        pts = [
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_FRONT
        ]
        res = list()
        for pt in pts:
            angle = self._state.angle - self._state.wheel_angle
            pt1 = Point.vector_from_angle(angle) * settings.VEHICLE.WHEEL_SIZE + pt
            pt2 = Point.vector_from_angle(angle + math.pi) * settings.VEHICLE.WHEEL_SIZE + pt
            res.append(pt1)
            res.append(pt2)
        return res

    def _check_collision(self, pos: Point, angle: np.float32, lines: List[Line]):
        bounds_lines = get_lines(self._get_bounds(pos, angle))
        for bl in bounds_lines:
            for l in lines:
                b, v = Line.distance_line_line(pos, np.float32(0), bl, l)
                if b and math.fabs(v) < 1e-1:
                    return True
        return False

    def _move_linear(self, t: np.float32, lines: List[Line]):
        run_len = self._state.velocity * t
        pos = self._state.pos + Point.vector_from_angle(self._state.angle)*run_len
        if self._check_collision(pos, self._state.angle, lines):
            return False
        self._state.pos = pos
        return True

    def _move_with_turn(self, t: np.float32, lines: List[Line]):
        run_len = self._state.velocity * t
        if np.allclose(run_len, np.float32(0.0), np.float32(1e-5)):
            return True
        abpt = (settings.VEHICLE.AXLE_BACK[0] + settings.VEHICLE.AXLE_BACK[1]) * 0.5 + self._state.pos
        afpt = (settings.VEHICLE.AXLE_FRONT[0] + settings.VEHICLE.AXLE_FRONT[1]) * 0.5 + self._state.pos
        al = (abpt - afpt).length()
        rb = np.float32(al / math.tan(math.fabs(self._state.wheel_angle)))
        beta = run_len/rb
        beta = np.float32(-beta if self._state.wheel_angle >= 0 else beta)
        move_angle = (self._state.angle + beta*0.5)
        delta_move = Point.vector_from_angle(move_angle, r=run_len)
        pos = self._state.pos + delta_move
        angle_new = self._state.angle + beta
        if self._check_collision(pos, angle_new, lines):
            return False
        self._state.pos = pos
        self._state.angle = angle_new
        return True

    def get_drive_time(self):
        return self._drive_time

    def step(self, t, lines: List[Line]):
        if t > 0:
            self._drive_time += t
        res1, res2 = self._step(t, lines)
        if res1:
            nv = self._state.to_reward_values()
            nv[settings.REWARD_KEY_COLLISION] = not res2
            nv[settings.REWARD_KEY_DRIVE_TIME] = self._drive_time
            reward = self._reward.get_reward(self._vehicle_id, nv)
        else:
            reward = None
        return res1 and res2, reward

    def model_ops(self, ops, t):
        """
        this functionality could be implemented with "dream" neural network.
        this implementation just for speed optimization.
        """
        s = self.state
        v = Vehicle(name=self._name + "_model_ops")
        dt = v.get_drive_time()
        res = list()
        for cmd in ops:
            v.state = s
            v._drive_time = dt
            v.commands(cmd)
            v.step(t, [])  # do not check for collision in dream functionality
            angle_norm, velocifty_norm = v.get_last_operation_info()
            if not np.allclose(velocifty_norm, np.float32(0.0),
                               settings.VEHICLE.TRANSMISSION_SPEED[-1]*settings.VEHICLE.SMALL_SPEED_COEF
                               ):
                res.append((angle_norm, velocifty_norm))
            else:
                v._drive_time = dt
                v._set_small_velocity()
                v.setup_reward()
                v.step(t, [])
                angle_norm, _ = v.get_last_operation_info()
                res.append((angle_norm, velocifty_norm))
        return res

    def _step(self, t, lines: List[Line]):
        """
        to process one time step -- one moment change for speed, rotation, position
        :param t: time interval
        :return: if step was ok
        """
        t = np.float32(t)
        if t <= 0:
            return False, False
        self._state.add(t)
        if not self._velocity_step(t):
            return False, False
        if not self._turn_step(t):
            return False, False
        if np.allclose(self._state.wheel_angle, 0.0, 1e-2):
            return True, self._move_linear(t, lines)
        else:
            return True, self._move_with_turn(t, lines)

    def get_input(self, env: StaticEnv):
        all_lines = env.get_lines()
        inputs = list()
        for index in settings.VEHICLE_STATE_INDEXES:
            inp = list()
            p = self._state.get_at(index)
            pt, angle0 = copy.deepcopy(p.pos), p.angle
            for ai in range(0, 360, settings.ANGLE_FOR_INPUT):
                angle = np.float32(math.pi*ai/180.0 + angle0)
                dist = [Line.distance_line_pt(pt, angle, l, pt) for l in all_lines]
                dist = [np.float32(math.fabs(d[1])) for d in dist if d[0]]
                dist.append(settings.MAX_INPUT_LINE_LENGTH)
                lmin = min(dist)
                pt2 = Point.vector_from_angle(angle, lmin) + pt
                inp.append(Line(pt, pt2))
            inputs.append(inp)
        return inputs

    def get_input_coefs_from_inputs(self, inputs):
        return [
                [   np.float32(
                        math.log(l.length - settings.MIN_INPUT_LINE_LENGTH + np.float32(1.0))
                        /math.log(settings.MAX_INPUT_LINE_LENGTH - settings.MIN_INPUT_LINE_LENGTH + np.float32(1.0))
                        - 0.5)
                    for l in inp
                ]
                for inp in inputs
            ]

    def get_input_coefs(self, env: StaticEnv):
        return self.get_input_coefs_from_inputs(self.get_input(env))

    def get_sensor_inputs(self):
        si = np.zeros((settings.SENSOR_INPUTS_SIZE,), dtype=np.float32)
        si[0] = np.float32(
            (self._state.velocity - settings.VEHICLE.TRANSMISSION_SPEED[0])
            / (settings.VEHICLE.TRANSMISSION_SPEED[-1] - settings.VEHICLE.TRANSMISSION_SPEED[0])
            - 0.5
        )
        si[1] = np.float32(
            (self._state.acceleration - settings.VEHICLE.TRANSMISSION_ACCELERATION[0])
            / (settings.VEHICLE.TRANSMISSION_ACCELERATION[-1] - settings.VEHICLE.TRANSMISSION_ACCELERATION[0])
            - 0.5
        )
        si[2] = np.float32(
            (self.wheel_angle - settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0])
            / (settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1] - settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0])
            - 0.5
        )
        si[3] = np.float32(
            (self.wheel_delta_angle - settings.VEHICLE.ROTATE_ACCELERATION[0])
            / (settings.VEHICLE.ROTATE_ACCELERATION[-1] - settings.VEHICLE.ROTATE_ACCELERATION[0])
            - 0.5
        )
        for i, index in enumerate(settings.VEHICLE_STATE_INDEXES):
            p = self._state.get_at(index)
            pt, angle0 = copy.deepcopy(p.pos), p.angle
            angle0 = math.fmod(angle0, math.pi*2.0)
            angle0 = math.fmod(angle0 + math.pi*2.0, math.pi*2.0)
            si[4 + 3*i + 0] = np.float32(angle0/(math.pi*2.0) - 0.5)
            si[4 + 3*i + 1] = np.float32(-0.5 if pt.y < 0 else 0.5)
            pt *= np.float32(1.0/settings.TARGET.TARGET_POINT_MAX_DISTANCE)
            if pt.length() > 1:
                pt *= np.float32(1.0/pt.length())
            si[4 + 3*i + 2] = np.float32(pt.x * 0.5)

        return si

    def get_last_operation_info(self):
        p_current = self._state.get_at(0)
        p_prev = self._state.get_at(1)
        current_angle = math.fmod(p_current.angle, np.float32(math.pi*2.0))
        prev_angle = math.fmod(p_prev.angle, np.float32(math.pi*2.0))
        current_angle = math.fmod(current_angle + np.float32(math.pi*2.0), np.float32(math.pi*2.0))
        prev_angle = math.fmod(prev_angle + np.float32(math.pi*2.0), np.float32(math.pi*2.0))
        sign = np.float32(1.0 if current_angle - prev_angle >= 0 else -1.0)
        delta_angle = angle_diff(current_angle, prev_angle)
        delta_angle = scipy.special.erf(delta_angle/np.sqrt(settings.ANGLE_FOR_OPERATION_MAX/25))
        delta_angle = delta_angle * sign
        delta_angle *= np.float32(0.5)  # [-0.5..0.5]
        velocity_norm = np.float32(
                (self._state.velocity - settings.VEHICLE.TRANSMISSION_SPEED[0])
                / (settings.VEHICLE.TRANSMISSION_SPEED[-1] - settings.VEHICLE.TRANSMISSION_SPEED[0])
                - 0.5)
        assert(delta_angle >= np.float32(-0.5) and delta_angle <= np.float32(0.5))
        assert(velocity_norm >= np.float32(-0.5) and velocity_norm <= np.float32(0.5))
        return delta_angle, velocity_norm

    pass  # class Vehicle
