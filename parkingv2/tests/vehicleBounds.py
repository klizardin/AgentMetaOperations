from env.vehicle import Vehicle
import RLParking.settings as settings

import numpy as np
import matplotlib.pyplot as plt
import math


def main():
    v = Vehicle("get_possible_delta_angles")
    max_velocity = settings.VEHICLE.TRANSMISSION_SPEED[-1]
    min_velocity = settings.VEHICLE.TRANSMISSION_SPEED[0]
    max_acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[-1]
    min_acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[0]
    max_wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1]
    min_wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0]
    max_wheel_delta_angle = settings.VEHICLE.ROTATE_ACCELERATION[-1]
    min_wheel_delta_angle = settings.VEHICLE.ROTATE_ACCELERATION[-1]
    count = 1000
    res = np.zeros((count, ), dtype=np.float32)
    res1 = np.zeros((count, ), dtype=np.float32)
    for i in range(count):
        s = v.state
        r = np.random.rand(4)
        s.velocity = r[0]*(max_velocity - min_velocity) + min_velocity
        s.acceleration = r[1]*(max_acceleration - min_acceleration) + min_acceleration
        s.wheel_angle = r[2]*(max_wheel_angle - min_wheel_angle) + min_wheel_angle
        s.wheel_delta_angle = r[3]*(max_wheel_delta_angle - min_wheel_delta_angle) + min_wheel_delta_angle
        v.state = s
        ops_list = v.get_next_available_ops()
        op_i = np.random.randint(0, len(ops_list))
        ops = ops_list[op_i]
        v.commands(ops)
        v.step(settings.VEHICLE_STEP_DURATION, list())
        delta_angle, velocity = v.get_last_operation_info()
        res[i] = delta_angle
        res1[i] = velocity
    #res[:] *= np.float32(2.0)
    #res[:] *= np.float32(180.0/math.pi)
    plt.hist(res)
    plt.show()


if __name__ == "__main__":
    main()