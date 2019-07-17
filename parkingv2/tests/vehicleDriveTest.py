from env.vehicle import Vehicle, StaticEnv, VehicleOp
from RLParking.db import StartCoordinates
from geom.primitives import Point, Line, Rect, between
from RLParking.settings import settings

import pygame
import numpy as np
import math


class KeyInfo:
    def __init__(self, key, cmd=None, *, pos=0, value=None):
        self.key = key
        self.cmd = cmd
        self.pos = pos
        self.value = value

    pass  # class KeyInfo


class GUITestVehicle:
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    SCREEN_CENTER = Point(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self._done = False
        self._vehicle = Vehicle("GUITestVehicle")
        self._env = StaticEnv()
        self.KEYS = [
            KeyInfo(pygame.K_3, VehicleOp.ACCELERATE_BACKWARD_2),
            KeyInfo(pygame.K_4, VehicleOp.ACCELERATE_BACKWARD_1),
            KeyInfo(pygame.K_5, VehicleOp.ACCELERATE_NONE),
            KeyInfo(pygame.K_6, VehicleOp.ACCELERATE_FORWARD_1),
            KeyInfo(pygame.K_7, VehicleOp.ACCELERATE_FORWARD_2),

            KeyInfo(pygame.K_x, VehicleOp.ROTATE_LEFT_2),
            KeyInfo(pygame.K_c, VehicleOp.ROTATE_LEFT_1),
            KeyInfo(pygame.K_v, VehicleOp.ROTATE_NONE),
            KeyInfo(pygame.K_b, VehicleOp.ROTATE_RIGHT_1),
            KeyInfo(pygame.K_n, VehicleOp.ROTATE_RIGHT_2),

            # KeyInfo(pygame.K_q, VehicleOp.FORWARD),
            # KeyInfo(pygame.K_a, VehicleOp.BACKWARD),

            KeyInfo(pygame.K_i, pos=1, value=True),

            KeyInfo(pygame.K_ESCAPE, pos=2, value=True),
        ]
        self._old_keys = [0] * len(self.KEYS)
        self._start_coordinates = StartCoordinates()
        self._initialize()

    def _initialize(self):
        sc = self._start_coordinates.get_rand()[0]
        self._vehicle.initialize(
            pos=sc.pos,
            angle=sc.angle,
            velocity=0.0
        )

    def _c(self, pt, c):
        return Point(pt.x*c.x, -pt.y*c.y) + self.SCREEN_CENTER

    def _draw_line(self, line: Line, coefs, color=(128, 128, 128)):
        pt0 = self._c(line.pt0, coefs)
        pt1 = self._c(line.pt1, coefs)
        pygame.draw.line(self._screen, color, (pt0.x, pt0.y), (pt1.x, pt1.y))

    def _draw_rect(self, rc: Rect, coefs, color=(128, 128, 128)):
        pt0 = self._c(rc.top_left, coefs)
        pt1 = self._c(rc.bottom_right, coefs)
        pygame.draw.rect(self._screen, color, (pt0.x, pt0.y, pt1.x-pt0.x, pt1.y-pt0.y))
        pass

    def _draw_input(self, coefs):
        inputs = self._vehicle.get_input(self._env)
        input_coefs = self._vehicle.get_input_coefs(self._env)
        for i, (l, c) in enumerate(zip(inputs[0], input_coefs[0])):
            self._draw_line(l, coefs)
            x1 = between(settings.INPUT_SHOW_RECT.left, settings.INPUT_SHOW_RECT.right, i / len(inputs[0]))
            x2 = between(settings.INPUT_SHOW_RECT.left, settings.INPUT_SHOW_RECT.right, (i+1) / len(inputs[0]))
            y = between(settings.INPUT_SHOW_RECT.top, settings.INPUT_SHOW_RECT.bottom, c + 0.5)
            rc = Rect(top_left=Point(x1, settings.INPUT_SHOW_RECT.top), bottom_right=Point(x2, y))
            self._draw_rect(rc, coefs)
        if settings.VISUALIZE_INPUT_HISTORY:
            for shift, inp_coefs in enumerate(input_coefs):
                xshift = shift*4/coefs.x
                for i, c in enumerate(inp_coefs):
                    x1 = between(settings.INPUT_SHOW_RECT.left, settings.INPUT_SHOW_RECT.right,
                                 i / len(inp_coefs)) + xshift
                    x2 = x1 + 4/coefs.x
                    y = between(settings.INPUT_SHOW_RECT.top, settings.INPUT_SHOW_RECT.bottom, c + 0.5)
                    rc = Rect(top_left=Point(x1, settings.INPUT_SHOW_RECT.top), bottom_right=Point(x2, y))
                    clr = 127*shift/4 + 128
                    self._draw_rect(rc, coefs, color=(clr, clr, clr))

    def _draw_dist_ltl(self, coefs):
        all_lines = self._env.get_lines()
        pt0 = self._vehicle.position
        angle = np.float32(math.pi*90/180)
        ll = 5
        l1 = Line(Point.vector_from_angle(angle, ll) + pt0, Point.vector_from_angle(angle + math.pi, ll) + pt0)
        angle = np.float32(math.pi * 10 / 180)
        dist = [Line.distance_line_line(pt0, angle, l1, l) for l in all_lines]
        dist = [d[1] for d in dist if d[0]]
        dist.append(np.float32(30))
        length = min(dist)
        pt1 = Point.vector_from_angle(angle, length) + pt0
        l2 = Line(pt0, pt1)
        self._draw_line(l1, coefs, color=(0, 128, 0))
        self._draw_line(l2, coefs, color=(0, 0, 128))

    def _draw_target(self, coefs):
        tl = np.float32(3.0)
        pt0 = Point.vector_from_angle(settings.TARGET.TARGET_ANGLE, tl) + settings.TARGET.TARGET_POINT
        pt1 = Point.vector_from_angle(settings.TARGET.TARGET_ANGLE + math.pi, tl) + settings.TARGET.TARGET_POINT
        l1 = Line(pt0, pt1)
        pt0 = Point.vector_from_angle(settings.TARGET.TARGET_ANGLE + math.pi*0.5, tl*0.5) + settings.TARGET.TARGET_POINT
        pt1 = Point.vector_from_angle(settings.TARGET.TARGET_ANGLE + math.pi*1.5, tl*0.5) + settings.TARGET.TARGET_POINT
        l2 = Line(pt0, pt1)
        self._draw_line(l1, coefs, (128, 0, 0))
        self._draw_line(l2, coefs, (0, 128, 0))

    def _draw_weights(self, coefs):
        for pos, w in self._start_coordinates.get_pos_with_weights():
            c1 = int((127+64) * w) + (128-64)
            self._draw_line(Line(pos, pos), coefs, (c1, c1, c1))

    def _get_lines(self):
        return self._env.get_lines()

    def _draw(self):
        coefs = Point(self.SCREEN_WIDTH/settings.SCREEN_SIZE.cx, self.SCREEN_HEIGHT/settings.SCREEN_SIZE.cy)

        if settings.VISUALIZE_START_POSITIONS_WEIGHTS:
            self._draw_weights(coefs)
        for l in self._env.get_lines():
            self._draw_line(l, coefs)

        self._draw_input(coefs)
        # self._draw_dist_ltl(coefs)
        self._draw_target(coefs)

        bounds = self._vehicle.get_bounds()
        axles = self._vehicle.get_axles()
        front_wheels = self._vehicle.get_front_wheels()
        pt0 = self._c(bounds[-1], coefs)
        for pt in bounds:
            pt1 = self._c(pt, coefs)
            pygame.draw.line(self._screen, (128, 128, 128), (pt0.x, pt0.y), (pt1.x, pt1.y))
            pt0 = pt1
        axles = [self._c(pt, coefs) for pt in axles]
        front_wheels = [self._c(pt, coefs) for pt in front_wheels]
        pygame.draw.line(self._screen, (128, 128, 128), (axles[0].x, axles[0].y), (axles[1].x, axles[1].y))
        pygame.draw.line(self._screen, (128, 128, 128), (axles[2].x, axles[2].y), (axles[3].x, axles[3].y))
        pygame.draw.line(self._screen, (128, 128, 128),
            (front_wheels[0].x, front_wheels[0].y),
            (front_wheels[1].x, front_wheels[1].y))
        pygame.draw.line(self._screen, (128, 128, 128),
            (front_wheels[2].x, front_wheels[2].y),
            (front_wheels[3].x, front_wheels[3].y))

    def _check_key(self, pressed):
        self._new_keys = [0] * len(self.KEYS)
        for i, k in enumerate(self.KEYS):
            self._new_keys[i] = 1 if pressed[k.key] else 0
        res = [None] * (max(self.KEYS, key=lambda x: x.pos).pos + 1)
        res[0] = list()
        for i, k in enumerate(self.KEYS):
            if self._new_keys[i] != 0 and self._old_keys[i] == 0:
                if k.cmd is not None:
                    res[0].append(k.cmd)
                else:
                    res[k.pos] = k.value
        self._old_keys = self._new_keys
        return res

    def run(self):
        self._initialize()
        t0 = pygame.time.get_ticks()
        while not self._done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True

            pressed = pygame.key.get_pressed()
            cmds, i, e = self._check_key(pressed)
            if e is not None:
                break
            elif i is not None:
                self._initialize()
                self._vehicle.setup_reward()
            elif cmds:
                self._vehicle.commands(cmds)
            else:
                self._vehicle.setup_reward()

            t1 = pygame.time.get_ticks()
            dt = (t1 - t0)/1000.0
            lines = self._get_lines()
            last_cmd_ok, reward = self._vehicle.step(dt, lines)
            if not last_cmd_ok and settings.VEHICLE.VERBOSE:
                print("failed to move (may be collision)")
                self._vehicle.reset()
            t0 = t1

            self._screen.fill((0, 0, 0))
            self._draw()

            pygame.display.flip()
            self._clock.tick(30)

    pass  # class GUITestVehicle


def main():
    # print("{0} -- {1}".format(PATHES.get_temp(),PATHES.is_temp_exist()))
    gui_vehicle = GUITestVehicle()
    gui_vehicle.run()


if __name__ == "__main__":
    main()
