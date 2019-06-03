from RLParking.settings import *
from env.vehicle import VehicleState, Vehicle, VehicleOperation, StaticEnv, VEHICLE_POSSIBLE_OPERATIONS
from RLParking.db import StartCoordinates, StartCoord
from geom.primitives import Point,Line,Rect, between
from RLParking.request import RequestType, Request
import pygame
import time


class AsyncVehicleBot:
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    SCREEN_CENTER = Point(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)

    def __init__(self, ui_bot:bool, start_coordinates: StartCoordinates, env: StaticEnv):
        self._ui_bot = ui_bot
        if self._ui_bot:
            pygame.init()
            self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self._clock = pygame.time.Clock()

        self._vehicle = Vehicle("UIBotVehicle" if self._ui_bot else "BotVehicle")
        self._done = False
        self._env = env
        self._start_coordinates = start_coordinates
        self._initialize()

    def _initialize(self):
        #print("start vehicle {} -- {}".format(self._vehicle.vehicle_id, self._vehicle.name))
        sc = self._start_coordinates.get_rand()[0]
        self._vehicle.initialize(
            pos=sc.pos,
            angle=sc.angle,
            velocity=0.0
        )
        self._inputs = self._vehicle.get_input(self._env)

    def _get_lines(self):
        return self._env.get_lines()

    def _c(self, pt, c):
        return Point(pt.x*c.x, -pt.y*c.y) + self.SCREEN_CENTER

    def _draw_line(self, line: Line, coefs, color = (128, 128, 128)):
        if not self._ui_bot:
            return
        pt0 = self._c(line.pt0, coefs)
        pt1 = self._c(line.pt1, coefs)
        pygame.draw.line(self._screen, color, (pt0.x, pt0.y), (pt1.x, pt1.y))

    def _draw_rect(self, rc: Rect, coefs, color = (128,128,128)):
        if not self._ui_bot:
            return
        pt0 = self._c(rc.top_left, coefs)
        pt1 = self._c(rc.bottom_right, coefs)
        pygame.draw.rect(self._screen, color, (pt0.x, pt0.y, pt1.x-pt0.x, pt1.y-pt0.y))
        pass

    def _draw_input(self, coefs):
        if not self._ui_bot:
            return
        inputs = self._inputs
        input_coefs = self._vehicle.get_input_coefs_from_inputs(inputs)
        for i,(l,c) in enumerate(zip(inputs[0], input_coefs[0])):
            self._draw_line(l, coefs)
            x1 = between(INPUT_SHOW_RECT.left, INPUT_SHOW_RECT.right, i / len(inputs[0]))
            x2 = between(INPUT_SHOW_RECT.left, INPUT_SHOW_RECT.right, (i+1) / len(inputs[0]))
            y = between(INPUT_SHOW_RECT.top, INPUT_SHOW_RECT.bottom, c + 0.5)
            rc = Rect(
                top_left=Point(x1, INPUT_SHOW_RECT.top)
                , bottom_right=Point(x2,y)
            )
            self._draw_rect(rc, coefs)
        if VISUALIZE_INPUT_HISTORY:
            for shift,inp_coefs in enumerate(input_coefs):
                xshift = shift*4/coefs.x
                for i, c in enumerate(inp_coefs):
                    x1 = between(INPUT_SHOW_RECT.left, INPUT_SHOW_RECT.right, i / len(inp_coefs)) + xshift
                    x2 = x1 + 4/coefs.x
                    y = between(INPUT_SHOW_RECT.top, INPUT_SHOW_RECT.bottom, c + 0.5)
                    rc = Rect(
                        top_left=Point(x1, INPUT_SHOW_RECT.top)
                        , bottom_right=Point(x2,y)
                    )
                    clr = 127*shift/4 + 128
                    self._draw_rect(rc, coefs, color=(clr,clr,clr))

    @staticmethod
    def _has(all_op_lists, op_list):
        for i, op_l in enumerate(all_op_lists):
            if len(op_l) != len(op_list):
                continue
            if all([op_1.value==op_2.value for op_1, op_2 in zip(op_l,op_list)]):
                return i
        return -1

    def _draw_net_results(self, ops, ops_values, coefs):
        net_results_size = len(VEHICLE_POSSIBLE_OPERATIONS)
        for i, possiblie_ops in enumerate(VEHICLE_POSSIBLE_OPERATIONS):
            vi = AsyncVehicleBot._has(ops, possiblie_ops)
            v = np.float32(0.0) if vi < 0 else ops_values[vi]
            v = np.clip(v, np.float32(-1.0), np.float32(1.0))
            x1 = between(NET_SHOW_RECT.left, NET_SHOW_RECT.right, i / (net_results_size + 1))
            x2 = between(NET_SHOW_RECT.left, NET_SHOW_RECT.right, (i+1) / (net_results_size + 1))
            y1 = between(NET_SHOW_RECT.top, NET_SHOW_RECT.bottom, v*0.8 + 0.5)
            y2 = between(NET_SHOW_RECT.top, NET_SHOW_RECT.bottom, 0.5)
            y1, y2 = min((y1,y2)), max((y1,y2))
            rc = Rect(
                top_left=Point(x1, y1)
                , bottom_right=Point(x2,y2)
            )
            if math.fabs(v) >= 0.5:
                clr = (128, 0, 0)
            else:
                clr = (128, 128, 128)
            self._draw_rect(rc, coefs, clr)
        x1 = between(NET_SHOW_RECT.left, NET_SHOW_RECT.right, net_results_size / (net_results_size + 1))
        x2 = between(NET_SHOW_RECT.left, NET_SHOW_RECT.right, (net_results_size + 1) / (net_results_size + 1))
        rc = Rect(
            top_left=Point(x1, NET_SHOW_RECT.top)
            , bottom_right=Point(x2, NET_SHOW_RECT.bottom)
        )
        self._draw_rect(rc, coefs)


    def _draw_dist_ltl(self, coefs):
        if not self._ui_bot:
            return
        all_lines = self._env.get_lines()
        pt0 = self._vehicle.position
        angle = np.float32(math.pi*90/180)
        ll = 5
        l1 = Line(Point.vector_from_angle(angle, ll) + pt0, Point.vector_from_angle(angle + math.pi, ll) + pt0)
        angle = np.float32(math.pi * 10 / 180)
        dist = [Line.distance_line_line(pt0, angle, l1, l) for l in all_lines]
        dist = [d[1] for d in dist if d[0]]
        dist.append(np.float32(30))
        len = min(dist)
        pt1 = Point.vector_from_angle(angle, len) + pt0
        l2 = Line(pt0,pt1)
        self._draw_line(l1, coefs, color=(0, 128, 0))
        self._draw_line(l2, coefs, color=(0, 0, 128))

    def _draw_target(self, coefs):
        if not self._ui_bot:
            return
        tl = np.float32(3.0)
        pt0 = Point.vector_from_angle(TARGET_ANGLE, tl) + TARGET_POINT
        pt1 = Point.vector_from_angle(TARGET_ANGLE + math.pi, tl) + TARGET_POINT
        l1 = Line(pt0, pt1)
        pt0 = Point.vector_from_angle(TARGET_ANGLE + math.pi*0.5, tl*0.5) + TARGET_POINT
        pt1 = Point.vector_from_angle(TARGET_ANGLE + math.pi*1.5, tl*0.5) + TARGET_POINT
        l2 = Line(pt0, pt1)
        self._draw_line(l1, coefs, (128, 0, 0))
        self._draw_line(l2, coefs, (0, 128, 0))

    def _draw_weights(self, coefs):
        if not self._ui_bot:
            return
        for pos, w in self._start_coordinates.get_pos_with_weights():
            c1 = int((127+64) * w) + (128-64)
            self._draw_line(Line(pos, pos), coefs, (c1,c1,c1))
        pass

    def _draw_vehicle(self, coefs):
        bounds = self._vehicle.get_bounds()
        axles = self._vehicle.get_axles()
        front_wheels = self._vehicle.get_front_wheels()
        pt0 = self._c(bounds[-1],coefs)
        for pt in bounds:
            pt1 = self._c(pt, coefs)
            pygame.draw.line(self._screen, (128,128,128), (pt0.x,pt0.y), (pt1.x, pt1.y))
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

    def _draw(self, ops, ops_values):
        if not self._ui_bot:
            return

        self._screen.fill((0, 0, 0))

        coefs = Point(self.SCREEN_WIDTH/SCREEN_SIZE.cx, self.SCREEN_HEIGHT/SCREEN_SIZE.cy)
        if VISUALIZE_START_POSITIONS_WEIGHTS:
            self._draw_weights(coefs)
        for l in self._env.get_lines():
            self._draw_line(l, coefs)
        self._draw_input(coefs)
        # self._draw_dist_ltl(coefs)
        self._draw_target(coefs)
        self._draw_vehicle(coefs)
        self._draw_net_results(ops, ops_values, coefs)

        pygame.display.flip()


    def process(self):
        try:
            self._ops = []
            self._ops_value = np.zeros((1,),dtype=np.float32)
            self._initialize()
            t0 = time.time()
            while True:
                yield from self._process_main()
                if self._ui_bot:
                    t1 = time.time()
                    while (
                        int(t1/1.5) # *VEHICLE_UI_BOT_STEPS_PER_SECOND
                        == int(t0/1.5) # *VEHICLE_UI_BOT_STEPS_PER_SECOND
                        ):
                        _ = yield Request(RequestType.NOP)
                        t1 = time.time()
                    t0 = t1
                    self._draw(self._ops, self._ops_values)
                pass
        except StopIteration as e:
            return

    def _set_operations_values(self, ops, ops_values):
        p = np.copy(ops_values.flatten())
        p -= np.min(p) - np.float32(0.05)
        s = np.sum(p)
        if s > 0: p /= s
        else: p[:] = np.float32(1.0/p.shape[0])
        indexes = np.random.choice(p.shape[0], min((STATE_LEARN_VALUES_COUNT,p.shape[0])), replace=False, p=p)
        values = np.zeros(indexes.shape, dtype=np.float32)
        selected_ops = list()
        for i in range(indexes.shape[0]):
            index = indexes[i]
            v = Vehicle("test_values")
            v.state = self._vehicle.state
            steps = 0
            op = ops[index]
            selected_ops.append(op)
            v.commands(op)
            last_op_ok, reward = v.step(t=VEHICLE_STEP_DURATION, lines=self._env.get_lines())
            final_state = not ((not reward[1]) and last_op_ok)
            reward_value = reward[0]
            steps += 1
            while not final_state: # and (reward_value == np.float32(0.0))
                ops1 = v.get_next_available_ops()
                inputs = v.get_input_coefs(self._env)
                inputs = np.array(inputs, dtype=np.float32)
                sensor_inputs = v.get_sensor_inputs()
                inputs = np.concatenate((inputs.flatten(), sensor_inputs))
                request = yield Request(
                    type=RequestType.GET_BEST_OPERATION,
                    inputs=inputs,
                    ops=ops1
                )
                op = ops1[request.get_best_operation()]
                v.commands(op)
                last_op_ok, reward = v.step(t=VEHICLE_STEP_DURATION, lines=self._env.get_lines())
                final_state = not ((not reward[1]) and last_op_ok)
                reward_value = reward[0]
                steps += 1
            values[i] = np.float32(math.pow(RL_RO_COEF,steps)*reward_value)
        if not selected_ops:
            return None, None
        else:
            return selected_ops, values

    def _process_main(self):
        ops = self._vehicle.get_next_available_ops()
        self._inputs = self._vehicle.get_input(self._env)
        inputs = self._vehicle.get_input_coefs_from_inputs(self._inputs)
        inputs = np.array(inputs, dtype=np.float32)
        sensor_inputs = self._vehicle.get_sensor_inputs()
        inputs = np.concatenate((inputs.flatten(),sensor_inputs))
        request = yield Request(
            type = RequestType.GET_BEST_OPERATION,
            inputs = inputs,
            ops = ops
        )

        if not self._ui_bot and (np.random.rand(1)[0] <= SET_OPERATION_VALUES_PROB):
            selected_ops, selected_ops_values = yield from self._set_operations_values(ops, request.results)
        else:
            selected_ops, selected_ops_values = None, None
        if selected_ops: op = selected_ops[np.argmax(selected_ops_values)]
        else: op = ops[request.get_best_operation()]

        self._ops = ops
        self._ops_values = request.results
        assert(len(self._ops) == self._ops_values.shape[0])
        # setup commands
        self._vehicle.commands(op)
        state = self._vehicle.state
        # run commands
        last_op_ok, reward = self._vehicle.step(t=VEHICLE_STEP_DURATION, lines=self._env.get_lines())
        final_state = not ((not reward[1]) and last_op_ok)
        _ = yield Request(
            type = RequestType.SAVE_TRAIN_STATE,
            state = state,
            next_state = self._vehicle.state,
            reward = reward[0],
            ops = selected_ops,
            values = selected_ops_values,
            final_state = final_state
        )
        if self._ui_bot:
            self._draw(self._ops, self._ops_values)
        if not ((not reward[1]) and last_op_ok):
            self._initialize()

    pass # class AsyncVehicleBot