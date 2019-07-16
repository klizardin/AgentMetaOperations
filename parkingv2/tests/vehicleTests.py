from env.vehicle import Vehicle, VehicleOp
from geom.primitives import Point
from RLParking.settings import settings

import unittest
import math
import numpy as np


settings.VEHICLE.VERBOSE = False


class VehicleTestCase(unittest.TestCase):
    def test_operations(self):
        v = Vehicle("V1")
        self.assertEqual(v.front, True)
        self.assertEqual(v.get_backward_count(), 2)
        self.assertEqual(v.get_acceleration_count(), 2)
        self.assertEqual(v.get_rotate_count(), settings.VEHICLE.ROTATE_ACCELERATION.shape[0] // 2)
        # self.assertTrue(v.command_transmission(True))
        # self.assertEqual(v.front, True)
        # self.assertTrue(v.command_transmission(False))
        # self.assertEqual(v.front, False)
        for i in range(v.get_backward_count()+1):
            self.assertTrue(v.command_backward(i))
        self.assertFalse(v.command_backward(-1))
        self.assertFalse(v.command_backward(v.get_backward_count() + 1))
        for i in range(v.get_acceleration_count()+1):
            self.assertTrue(v.command_accelerate(i))
        self.assertFalse(v.command_accelerate(-1))
        self.assertFalse(v.command_accelerate(v.get_acceleration_count() + 1))
        for i in range(-v.get_rotate_count(), v.get_rotate_count() + 1):
            self.assertTrue(v.command_turn(i))
        self.assertFalse(v.command_turn(-v.get_rotate_count() - 1))
        self.assertFalse(v.command_turn(v.get_rotate_count() + 1))

    def test_step_errors(self):
        v = Vehicle("V1")
        res, _ = v.step(-0.1, [])
        self.assertFalse(res)
        res, _ = v.step(0, [])
        self.assertFalse(res)
        v.commands([VehicleOp.NONE, ])
        res, _ = v.step(0.1, [])
        self.assertTrue(res)

    def test_step_velocity(self):
        v = Vehicle("V1")
        long_time = 1000.0
        v.commands([VehicleOp.NONE, ])
        v.command_accelerate(v.get_acceleration_count())
        v.step(long_time, [])
        self.assertAlmostEqual(v.velocity, settings.VEHICLE.TRANSMISSION_SPEED[-1], np.float32(1e-2))
        # v.commands([VehicleOp.NONE, ])
        # v.command_backward(v.get_backward_count())
        # v.step(long_time, [])
        # self.assertAlmostEqual(v.velocity, 0.0, 1e-2)
        # self.assertTrue(v.command_transmission(False))
        v.commands([VehicleOp.NONE, ])
        v.command_backward(v.get_backward_count())
        v.step(long_time, [])
        self.assertAlmostEqual(v.velocity, settings.VEHICLE.TRANSMISSION_SPEED[0], np.float32(1e-2))
        # v.commands([VehicleOp.NONE, ])
        # v.command_backward(v.get_backward_count())
        # v.step(long_time, [])
        # self.assertAlmostEqual(v.velocity, 0.0, 1e-2)

    def test_step_turn(self):
        v = Vehicle("V1")
        long_time = 1000.0
        v.commands([VehicleOp.NONE, ])
        v.command_turn(-v.get_rotate_count())
        v.step(long_time, [])
        self.assertAlmostEqual(v.wheel_angle, settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0], np.float32(1e-2))
        v.commands([VehicleOp.NONE, ])
        v.command_turn(v.get_rotate_count())
        v.step(long_time, [])
        self.assertAlmostEqual(v.wheel_angle, settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1], np.float32(1e-2))

    def test_get_bounds(self):
        v = Vehicle("V1")
        pos = Point(10, 10)
        v.initialize(pos, math.pi*0.5, 0.0)
        bounds = v.get_bounds()
        self.assertEqual(bounds[0], pos + Point(-1, 2))
        self.assertEqual(bounds[1], pos + Point(1, 2))
        self.assertEqual(bounds[2], pos + Point(1, -2))
        self.assertEqual(bounds[3], pos + Point(-1, -2))


if __name__ == '__main__':
    unittest.main()
