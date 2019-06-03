import time

import numpy as np
import pygame
import copy

from game.tetris import Operation, Tetris, FieldState, game_metrics
from mc.config import *
from mc.request import RequestType, Request


class TetrisPositionInfo:
    def __init__(self, x, y, r, gravity, reward):
        self.x = x
        self.y = y
        self.r = r
        self.gravity = gravity
        self.reward = reward
        self.parents = set()
        self.childs = set()
        self.possible_ops = set()
        self.value = np.float32(0.0)
        self.finit_state = False
        self.start_state = False

    def has_reward(self):
        return (self.reward != 0
            or self.finit_state or self.start_state
            )

    def get_reward(self):
        if self.finit_state:
            return -1
        if self.start_state:
            return 1
        else:
            return self.reward / MAX_FIGURE_SIZE

    def is_next_to(self, x, y, r, rotate_positions):
        dr = abs(self.r - r)
        if (rotate_positions > 1) and (dr == rotate_positions - 1):
            dr = 1
        return y - self.y == 1  \
            and abs(self.x - x) + dr == 1

    def is_same(self, pos):
        return self.x==pos.x    \
            and self.y==pos.y   \
            and self.r==pos.r

    def fill_possible_ops(self, rotate_positions):
        for children in self.childs:
            dr = children.r - self.r
            if (rotate_positions > 1) and abs(dr) == rotate_positions - 1:
                dr = 1 if dr < 0 else -1
            if dr < 0:
                self.possible_ops.add(Operation.ROTATE_UCW)
                continue
            elif dr > 0:
                self.possible_ops.add(Operation.ROTATE_CW)
                continue
            dx = children.x - self.x
            if dx < 0:
                self.possible_ops.add(Operation.MOVE_LEFT)
                continue
            elif dx > 0:
                self.possible_ops.add(Operation.MOVE_RIGHT)
                continue
            self.possible_ops.add(Operation.NOP)

    def compact(self, illegal_pos: bool = False):
        if not self.possible_ops:
            self.possible_ops = None
        if not self.parents or illegal_pos:
            self.parents = None
        self.childs = None
        if self.parents:
            self.parents = list(self.parents)
        if self.possible_ops:
            self.possible_ops = list(self.possible_ops)

    def current_y(self):
        return self.y

    def next_y(self):
        return self.y - 1

    def next_x(self):
        return self.x + 1

    def prev_x(self):
        return self.x - 1

    def next_r(self, rotate_positions):
        return (self.r + 1) % rotate_positions

    def prev_r(self, rotate_positions):
        r = self.r - 1
        if r < 0:
            r += rotate_positions
        return r

    pass #TestPositionInfo

def get_tetris_end_positions(src_tetris: Tetris):
    pts = list()
    for y in range(FIELD_FULL_HEIGHT):
        for x in range(FIELD_WIDTH):
            for r in range(src_tetris.figure_positions):
                x1,y1,r1 = x,y,r
                tetris = src_tetris.copy()
                if not tetris.test_figure_pos(x1,y1,r1):
                    continue
                if not tetris.set_figure_pos(x1,y1,r1):
                    continue
                g = tetris.gravity()
                if not g:
                    tetris.add_figure()
                    reward = tetris.cut_lines()
                else:
                    reward = 0
                pos = TetrisPositionInfo(x1, y1, r1, g, reward)
                pts.append(pos)
    end_positions = [pt for pt in pts if not pt.gravity]
    for pt in end_positions:
        pt.compact()
    return end_positions

def get_tetris_positions(src_tetris: Tetris):
    pts = list()
    illegal = list()
    positions = []
    positions.append([
        [None for r in range(src_tetris.figure_positions)]
        for x in range(FIELD_WIDTH + 2)
    ])

    for y in range(FIELD_FULL_HEIGHT):
        positions.append([
            [None for r in range(src_tetris.figure_positions)]
            for x in range(FIELD_WIDTH + 2)
        ])
        for x in range(FIELD_WIDTH):
            for r in range(src_tetris.figure_positions):
                x1,y1,r1 = x,y,r
                tetris = src_tetris.copy()
                if not tetris.test_figure_pos(x1,y1,r1):
                    continue
                if not tetris.set_figure_pos(x1,y1,r1):
                    illegal.append(TetrisPositionInfo(x1, y1, r1, False, 0))
                    continue
                g = tetris.gravity()
                if not g:
                    tetris.add_figure()
                    reward = tetris.cut_lines()
                else:
                    reward = 0
                pos = TetrisPositionInfo(x1, y1, r1, g, reward)
                pts.append(pos)
                positions[y1+1][x1+1][r1] = pos

    #if not [pt for pt in pts if not pt.gravity]:
    #    start_position_info, all_positions, end_positions, illegal

    start_pos, start_rotate = src_tetris.get_figure_position(), src_tetris.get_figure_rotate()
    start_position_info = TetrisPositionInfo(start_pos.x, start_pos.y, start_rotate, True, 0)
    all_positions = set()
    all_positions.add(start_position_info)
    pts_neigbors = [start_position_info]
    while pts_neigbors:
        pts_neigbors_new_all = set()
        for nieghbor in pts_neigbors:
            pts_neigbors_new = [
                positions[nieghbor.current_y()+1][nieghbor.x+1][nieghbor.next_r(src_tetris.figure_positions)],
                positions[nieghbor.current_y()+1][nieghbor.x+1][nieghbor.prev_r(src_tetris.figure_positions)],
                positions[nieghbor.current_y()+1][nieghbor.next_x()+1][nieghbor.r],
                positions[nieghbor.current_y()+1][nieghbor.prev_x()+1][nieghbor.r],
                positions[nieghbor.current_y()+1][nieghbor.x+1][nieghbor.r],
            ]
            pts_neigbors_new = [pt
                for pt in pts_neigbors_new
                if pt is not None
                ]
            if not pts_neigbors_new:
                continue
            pts_neigbors_new = [
                positions[pt.next_y()+1][pt.x+1][pt.r]
                for pt in pts_neigbors_new
            ]
            pts_neigbors_new = [pt
                for pt in pts_neigbors_new
                if (pt is not None) and (pt not in all_positions)
                ]
            if not pts_neigbors_new:
                continue
            for pt in pts_neigbors_new:
                nieghbor.childs.add(pt)
            for pt in pts_neigbors_new:
                pts_neigbors_new_all.add(pt)

        pts_neigbors = [pt
            for pt in pts_neigbors_new_all
            if pt not in all_positions
            ]
        for pt in pts_neigbors:
            all_positions.add(pt)

    all_positions = list(all_positions)
    end_positions = [pt for pt in all_positions if not pt.gravity]
    if not end_positions:
        for pt in all_positions:
            pt.fill_possible_ops(src_tetris.figure_positions)
            pt.compact()
        for pt in illegal:
            pt.compact(True)
        return start_position_info, all_positions, all_positions, illegal

    for ep in end_positions:
        t1 = src_tetris.copy()
        assert(t1.set_figure_pos(ep.x, ep.y, ep.r))
        assert(not t1.gravity())
        assert(t1.add_figure())
        for fi in range(src_tetris.figures_count):
            t2 = t1.copy()
            if ep.finit_state or ep.start_state:
                break
            if not t2.next_figure(fi):
                ep.finit_state = True
                break
            if t2.is_empty_field():
                ep.start_state = True
                break

    #all_positions = [
    #    pt
    #    for pt in all_positions
    #    if pt.childs or not pt.gravity
    #]
    for pt in all_positions:
        for children in pt.childs:
            if pt not in children.parents:
                children.parents.add(pt)

    for pt in all_positions:
        pt.fill_possible_ops(src_tetris.figure_positions)
        pt.compact()

    for pt in illegal:
        pt.compact(True)
    return start_position_info, all_positions, end_positions, illegal


class BotGame:
    """
    bot game class
    """
    def __init__(self, ui_mode = False):
        self._tetris = Tetris()
        self._ui_mode = ui_mode
        self._bot_id = game_metrics.get_next_tetris_id()
        if self._ui_mode:
            pygame.init()
            self._screen = pygame.display.set_mode((300, 700))
            self._warn_map = np.zeros((FIELD_FULL_HEIGHT,FIELD_WIDTH), dtype=np.float32)
        else:
            self._warn_map = None

    def _send_state_for_step(self, start_position_info, all_positions, end_positions, illegal):
        ts = self._tetris.state
        if not self._ui_mode:
            _ = yield Request(
                RequestType.SET_TRAIN_DATA,
                tetris_state=ts,
                start_position=start_position_info,
                all_positions=None, #all_positions,
                end_positions=end_positions,
                illegal_positions=illegal
            )
        else:
            ops = list()
            for pt in all_positions:
                tetris = self._tetris.copy()
                if not tetris.set_figure_pos(pt.x, pt.y, pt.r):
                    continue
                ops.append(tetris.get_op_info())

            tetris = self._tetris.copy()
            s1 = FieldState(tetris.get_field(), ops=ops)
            request = yield Request(
                RequestType.GET_BEST_OPERATION_BATCH,
                field_states = [s1]
            )

            self._warn_map[:,:] = np.float32(1.0)
            count = np.zeros(self._warn_map.shape, dtype=np.float32)
            for pt, op in zip(all_positions, request.field_states[0].ops):
                self._warn_map[pt.y, pt.x] += np.float32(op.value)
                count[pt.y, pt.x] += np.float32(1.0)
            count = np.where(count == 0, np.float32(1.0), count)
            self._warn_map /= count
            mask = self._warn_map == 1.0

            m1 = np.min(np.min(self._warn_map))
            self._warn_map = np.where(mask, np.float32(-1), self._warn_map)
            m2 = np.max(np.max(self._warn_map))
            self._warn_map = np.where(mask, np.float32(0.0), self._warn_map)
            self._warn_map -= m1
            if m2 - m1 > 0:
                self._warn_map /= (m2 - m1)*2
            self._warn_map += (m1 + np.float32(0.5))/2
            self._warn_map = np.clip(self._warn_map, np.float32(0.0), np.float32(1.0))
        return

    def _gravity(self):
        if self._tetris.gravity():
            return
        self._game_length += 1
        p_last = self._tetris.get_figure_position()
        r_last = self._tetris.get_figure_rotate()
        assert(self.ep.x == p_last.x and self.ep.y == p_last.y and self.ep.r == r_last)
        self._tetris.add_figure()
        reward = self._tetris.cut_lines()
        game_metrics.reward(self._bot_id, reward / MAX_FIGURE_SIZE)
        if ((not self._tetris.next_figure(np.random.randint(0, 100))) or (self._game_length > MAX_GAME_LENGTH)):
            game_metrics.new_game(self._bot_id)
            game_metrics.print_metrics()
            self._tetris.start()
            self._game_length = 0
            assert(self._tetris.next_figure(np.random.randint(0, 100)))

        self.start_position_info, self.all_positions, self.end_positions, self.illegal = \
            get_tetris_positions(self._tetris)
        yield from self._send_state_for_step(self.start_position_info, self.all_positions, self.end_positions,
                        self.illegal)
        yield from self._calc_ops()
        return

    def _draw(self):
        self._screen.fill((0, 0, 0))
        self._tetris.draw(pygame, self._screen, 10, 10, 280, 680, self._warn_map)
        pygame.display.flip()

    def _get_ops_info(self, end_positions):
        ops_info = list()
        for ep in end_positions:
            t1 = self._tetris.copy()
            assert (t1.set_figure_pos(ep.x, ep.y, ep.r))
            assert (not t1.gravity())
            ops_info.append(t1.get_op_info())
        return ops_info

    def _get_ops(self, end_positions, request: Request):
        ops = [(ep,op.value) for ep, op in zip(end_positions, request.field_state.ops)]
        op = max(ops, key=lambda x:x[1])
        ep = op[0]
        self.ep = copy.deepcopy(ep)
        ops = list()
        while ep.parents is not None:
            ops.append(copy.deepcopy(ep))
            ep = ep.parents[0]
        ops.append(copy.deepcopy(ep))
        ops[:] = ops[::-1]
        self.ops_pos = copy.deepcopy(ops)
        for i,op in enumerate(ops):
            if i == 0:
                continue
            ops[i-1].childs = [op]
            ops[i-1].possible_ops = set()
            ops[i-1].fill_possible_ops(self._tetris.figure_positions)
            ops[i-1].possible_ops = list(ops[i-1].possible_ops)
        ops[-1].possible_ops = [Operation.NOP]
        ops = [op.possible_ops[0] for op in ops]
        return ops

    def _calc_ops(self):
        field = self._tetris.get_field(False)
        ops_info = self._get_ops_info(self.end_positions)
        request = yield Request(
            RequestType.GET_BEST_OPERATION if self._ui_mode else RequestType.SEARCH_BEST_OPERATION,
            field_state=FieldState(field=field, ops=ops_info)
        )
        self.ops = self._get_ops(self.end_positions, request)
        self.op_index = 0

    def process(self):
        try:
            self._tetris.start()
            self._game_length = 0
            self._tetris.next_figure(np.random.randint(0,100))
            self.start_position_info, self.all_positions, self.end_positions, self.illegal \
                = get_tetris_positions(self._tetris)
            yield from self._send_state_for_step(self.start_position_info, self.all_positions, self.end_positions,
                            self.illegal)
            yield from self._calc_ops()
            t0 = time.time()
            while True:
                yield from self._process_main()
                if self._ui_mode:
                    t1 = time.time()
                    while int(t1*2) == int(t0*2):
                        _ = yield Request(RequestType.NOP)
                        t1 = time.time()
                    t0 = t1
                    self._draw()
                pass
        except StopIteration as e:
            return

    def _process_main(self):
        op = self.ops[self.op_index] if self.op_index < len(self.ops) else Operation.NOP
        self.op_index += 1
        if op.value == Operation.MOVE_LEFT.value:
            self._tetris.move_left()
        elif op.value == Operation.MOVE_RIGHT.value:
            self._tetris.move_right()
        elif op.value == Operation.ROTATE_CW.value:
            self._tetris.rotate_figure_cw()
        elif op.value == Operation.ROTATE_UCW.value:
            self._tetris.rotate_figure_ucw()
        elif op.value == Operation.NOP.value:
            pass
        game_metrics.next_step(self._bot_id)
        if self.op_index < len(self.ops_pos):
            ep = self.ops_pos[self.op_index]
            p_last = self._tetris.get_figure_position()
            r_last = self._tetris.get_figure_rotate()
            assert(ep.x == p_last.x and (ep.y+1 == p_last.y or ep.y == p_last.y) and ep.r == r_last)
        yield from self._gravity()

    pass # class BotGame
