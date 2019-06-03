"""
module to implement class for tetris game
"""

import copy
from enum import Enum
from typing import List

import numpy as np
import pygame

from mc.config import *


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    pass #class Position

class Operation(Enum):
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    ROTATE_CW = 3
    ROTATE_UCW = 4
    NOP = 5

    pass #class Operation

class OperationInfo:
    def __init__(self, op_arr, figure, figure_pos, op : Operation):
        self.op_arr = op_arr
        self.figure = figure
        self.figure_pos = figure_pos
        self.op = op
        self.value = np.float32(0.0)

    pass #class OperationInfo

class FieldState:
    def __init__(self, field, *,
                 op : OperationInfo = None,
                 ops : List[OperationInfo] = None,
                 value = 0.0,
                 search_step = False,
                 ):
        self.field = field
        self.op = op
        self.ops = ops
        self.value = np.float32(value)
        self.search_step = search_step

    def get_best_op(self):
        assert(self.ops is not None)
        i, _ = max(enumerate(self.ops), key = lambda x: x[1].value)
        return self.ops[i]

    pass #class FieldState

class TetrisState:
    def __init__(self, field, figure_index, figure_pos, figure_rotate, figure):
        self.field = np.copy(field)
        self.figure_index = figure_index
        self.figure_pos = copy.copy(figure_pos)
        self.figure_rotate = figure_rotate
        self.figure = np.copy(figure)

    pass #class TetrisState

class GameMetrics:
    def __init__(self):
        self._game_length = dict()
        self._game_figures_count = dict()
        self._game_reward = dict()
        self._tetris_id_last = 0
        self._avg_game_length = 100.0
        self._avg_game_reward = 0.0
        self._avg_game_figures_count = 10.0
        pass

    def initialize(self, avg_game_reward):
        self._avg_game_reward = avg_game_reward

    def get_next_tetris_id(self):
        self._tetris_id_last += 1
        return self._tetris_id_last

    def free_tetris_id(self, tetris):
        if tetris not in self._game_length:
            return
        del self._game_length[tetris]

    def next_step(self, tetris):
        if tetris in self._game_length:
            self._game_length[tetris] += 1
        else:
            self._game_length[tetris] = 1

    def reward(self, tetris, reward):
        if tetris in self._game_reward:
            self._game_reward[tetris] += reward
        else:
            self._game_reward[tetris] = reward
        if tetris in self._game_figures_count:
            self._game_figures_count[tetris] += 1
        else:
            self._game_figures_count[tetris] = 1

    def new_game(self, tetris):
        if tetris in self._game_length:
            self._avg_game_length += (self._game_length[tetris] - self._avg_game_length) * 1e-2
            self._game_length[tetris] = 0
        if tetris in self._game_reward:
            self._avg_game_reward += (self._game_reward[tetris] - self._avg_game_reward) * 1e-2
            self._game_reward[tetris] = 0
        if tetris in self._game_figures_count:
            self._avg_game_figures_count += (self._game_figures_count[tetris] - self._avg_game_figures_count) * 1e-2
            self._game_figures_count[tetris] = 0

    def print_metrics(self):
        if VERBOSE:
            print("avg(len(game))={0:.4} avg(game_figures)={2:.4} avg(game reward)={1:.4}"
                .format(
                    self._avg_game_length,
                    self._avg_game_reward,
                    self._avg_game_figures_count
                    )
                )

    pass #class GameMetrics

game_metrics = GameMetrics()

class Tetris:
    """
    class of the tetris game base methods
    """

    def __init__(self):
        """
        initialize class constant members
        """
        self._figures_shapes = (
            (
                (1, 1, 1),
                (0, 1, 0)
            ),
            (
                (0, 1, 1),
                (1, 1, 0)
            ),
            (
                (1, 1, 0),
                (0, 1, 1)
            ),
            (
                (1, 0, 0),
                (1, 1, 1)
            ),
            (
                (0, 0, 1),
                (1, 1, 1)
            ),
            (
                (1, 1, 1, 1)
            ),
            (
                (1, 1),
                (1, 1)
            )
        )
        self._figure_positions_all = (4,4,4,4,4,2,1)
        self._width = FIELD_WIDTH
        self._height = FIELD_HEIGHT
        self._field = None
        self._clear()
        return

    def _rotate(self, cwc : int):
        """
        to rotate figure
        :param cwc: -1|1 -- to rotate unti clock wise or clock wise
        :return: bool if can rotate
        """
        new_figure = np.rot90(self._figure, cwc)
        if self._test_figure(new_figure, self._figure_pos):
            self._figure = new_figure
            self._figure_rotate += cwc
            if self._figure_rotate < 0:
                self._figure_rotate += self._figure_positions
            self._figure_rotate %= self._figure_positions
            return True
        return False

    def _move(self, dx : int):
        """
        to move figure to the left or to the right
        :param dx: -1 -- move to left; 1 -- move to the right
        :return: bool if can move
        """
        figure_pos = Position(x = self._figure_pos.x, y = self._figure_pos.y)
        figure_pos.x += dx
        if self._test_figure(self._figure, figure_pos):
            self._figure_pos = figure_pos
            return True
        return False

    def get_possible_ops(self):
        result = list()
        if self._figure is None:
            return result

        figure, figure_pos = self._figure, self._figure_pos
        result.append(OperationInfo(None, self._figure, self._figure_pos, Operation.NOP))
        if self._move(-1):
            result.append(OperationInfo(None, self._figure,self._figure_pos, Operation.MOVE_LEFT))
            self._figure, self._figure_pos = figure, figure_pos
        if self._move(1):
            result.append(OperationInfo(None, self._figure,self._figure_pos, Operation.MOVE_RIGHT))
            self._figure, self._figure_pos = figure, figure_pos
        if self._rotate(-1):
            result.append(OperationInfo(None, self._figure,self._figure_pos, Operation.ROTATE_UCW))
            self._figure, self._figure_pos = figure, figure_pos
        if self._rotate(1):
            result.append(OperationInfo(None, self._figure,self._figure_pos, Operation.ROTATE_CW))
            self._figure, self._figure_pos = figure, figure_pos
        return result

    def _test_figure(self, figure, pos : Position):
        """
        test if new figure position of figuire rotation is correct
        :param figure: new figure
        :param pos: new position
        :return:
        """

        if figure is None:
            return False
        fw = figure.shape[1]
        fh = figure.shape[0]
        pos_tl = Position(x = pos.x - fw//2, y = pos.y - fh//2)
        if not ((pos_tl.x >= 0)
            and (pos_tl.x + fw <= self._width)
            and (pos_tl.y >= 0)
            and (pos_tl.y + fh <= self._height + MAX_FIGURE_SIZE)
        ):
            return False

        for y in range(fh):
            for x in range(fw):
                if (figure[y, x] > 0
                    and self._field[pos_tl.y + y, pos_tl.x + x] > 0
                ):
                    return False
        return True

    def _cut_line(self,field,y : int):
        """
        helpfull function to cut line of the field at the position y
        :param field: numpy array of field
        :param y: line position to cut off
        :return:
        """

        while y < self._height - 1:
            field[y, :] = field[y+1, :]
            y += 1
            pass
        field[self._height-1,:] = np.zeros((field.shape[1],),dtype=np.float32)
        return

    def next_figure(self, rv : int):
        """
        to get next random figure of the game
        :param rv: random integer number
        :return:
        """
        rv = int(rv) % len(self._figures_shapes)
        return self._set_figure(rv, Position(x = self._width // 2, y = self._height), 0)

    def _set_figure(self, figure_index, figure_pos, figure_rotate):
        self._figure_index = figure_index
        self._figure_pos = copy.copy(figure_pos)
        assert(len(self._figure_positions_all) == len(self._figures_shapes))
        fig = self._figures_shapes[figure_index]
        self._figure_positions = self._figure_positions_all[self._figure_index]
        w = h = len(fig)
        flat_array = isinstance(fig[0], int)
        if flat_array:
            h = 1
        else:
            w = len(fig[0])

        self._figure = np.zeros((h,w), dtype=np.float32)
        if flat_array:
            self._figure[0,:] = fig
        else:
            for y in range(h):
                self._figure[y,:] = fig[y]
        self._figure_rotate = figure_rotate
        if self._figure_rotate < 0:
            self._figure_rotate = (self._figure_rotate % self._figure_positions) + self._figure_positions
        self._figure_rotate %= self._figure_positions
        if not self._rotate(self._figure_rotate):
            self._clear()
            return False
        return True

    def _clear(self):
        self._figure_rotate = 0
        self._figure = None
        self._figure_pos = None
        self._figure_index = None
        self._figure_positions = 4
        self._figure_rotate = 0

    def copy(self):
        tetris = Tetris()
        tetris._figure = np.copy(self._figure)
        tetris._figure_pos = copy.copy(self._figure_pos)
        tetris._field = np.copy(self._field)
        tetris._figure_index = self._figure_index
        tetris._figure_positions = self._figure_positions
        return tetris

    @property
    def state(self):
        assert((self._figure is not None)
               and (self._figure_pos is not None)
               and (self._figure_index is not None)
               and (self._field is not None)
               )
        return TetrisState(self._field, self._figure_index, self._figure_pos, self._figure_rotate, self._figure)

    @state.setter
    def state(self, ts: TetrisState):
        self._field = np.copy(ts.field)
        if not self._set_figure(ts.figure_index, ts.figure_pos, ts.figure_rotate):
            self._clear()
            return False
        return True

    @property
    def figures_count(self):
        return len(self._figures_shapes)

    def get_figure_position(self):
        return copy.copy(self._figure_pos)

    def get_figure_rotate(self):
        return self._figure_rotate

    def is_empty_field(self):
        return np.all(self._field == np.float32(0.0))

    def test_figure_pos(self, x,y,r):
        assert (r < self._figure_positions)
        pos = Position(x, y)
        figure = np.rot90(self._figure, r)
        if figure is None:
            return False
        fw = figure.shape[1]
        fh = figure.shape[0]
        pos_tl = Position(x = pos.x - fw//2, y = pos.y - fh//2)
        if not ((pos_tl.x >= 0)
            and (pos_tl.x + fw <= self._width)
            and (pos_tl.y >= 0)
            and (pos_tl.y + fh <= self._height + MAX_FIGURE_SIZE)
        ):
            return False
        return True

    def set_figure_pos(self, x,y,r):
        assert(r<self._figure_positions)
        pos = Position(x,y)
        new_figure = np.rot90(self._figure, r)
        if self._test_figure(new_figure, pos):
            self._figure = new_figure
            self._figure_pos = pos
            return True
        return False

    def set_figure_illegal_pos(self, x,y,r):
        assert(r<self._figure_positions)
        pos = Position(x,y)
        new_figure = np.rot90(self._figure, r)
        self._figure = new_figure
        self._figure_pos = pos
        return True

    @property
    def figure_positions(self):
        return self._figure_positions

    def rotate_figure_cw(self):
        """
        to rotate figure clock wise
        :return: bool if can to rotate
        """
        return self._rotate(1)

    def rotate_figure_ucw(self):
        """
        to rotate figure unti clock wise
        :return: bool if can to rotate
        """
        return self._rotate(-1)

    def move_left(self):
        """
        to move figure left by one position
        :return: bool if can move left
        """
        return self._move(-1)

    def move_right(self):
        """
        to move figure right by one position
        :return: True or False if can move right
        """
        return self._move(1)

    def gravity(self):
        """
        implement figure fall down by one position
        :return: bool if can do next step
        """
        figure_pos = Position(x = self._figure_pos.x, y = self._figure_pos.y)
        figure_pos.y -= 1
        if self._test_figure(self._figure, figure_pos):
            self._figure_pos = figure_pos
            return True
        return False

    def add_figure(self):
        """
        add figure to the field after end of figure move
        :return: bool if can add Figure
        """
        if ((self._figure is None)
            or (not self._test_figure(self._figure, self._figure_pos))
        ):
            return False
        fw = self._figure.shape[1]
        fh = self._figure.shape[0]
        pos_tl =  Position(x = self._figure_pos.x - fw//2, y = self._figure_pos.y - fh//2)
        for x in range(fw):
            for y in  range(fh):
                if self._figure[y, x] > 0:
                    assert(self._field[pos_tl.y + y, pos_tl.x + x] == 0)
                    self._field[pos_tl.y + y, pos_tl.x + x] = self._figure[y, x]
        self._figure = None
        self._figure_pos = None
        return True

    @property
    def figure_line(self):
        if self._figure_pos is not None:
            fh = self._figure.shape[0]
            return self._figure_pos.y - fh // 2
        else:
            assert(False)
            return 0

    def cut_lines(self):
        """
        cut lines after figure adding
        :return: score
        """
        full_row = np.array([1]*self._width, dtype=np.float32)
        points = 0
        for y in range(self._field.shape[0]):
            while np.all(self._field[y,:] == full_row):
                self._cut_line(self._field, y)
                points += 1
        return points

    def start(self):
        """
        start new tetris game
        :return:
        """
        self._field = np.zeros((self._height + MAX_FIGURE_SIZE, self._width), dtype=np.float32)
        self._figure = None
        self._figure_pos = None
        return

    def _add_op_to_field(self,result, figure, figure_pos):
        """
        add operation data to the field
        :param result:
        :param figure: operation figure to add
        :param figure_pos: operation figure position to add
        :return:
        """
        if figure is not None:
            fw = figure.shape[1]
            fh = figure.shape[0]
            pos_tl =  Position(x = figure_pos.x - fw//2, y =  figure_pos.y - fh//2)
            for y in range(fh):
                for x in range(fw):
                    if figure[y, x] > 0:
                        assert(result[pos_tl.y + y, pos_tl.x + x] == 0)
                        result[pos_tl.y + y, pos_tl.x + x] = figure[y, x]
        return result

    def get_field(self, with_op: bool = True):
        """
        return field with current figure
        :return: field numpy array
        """
        result = np.zeros((FIELD_FULL_HEIGHT, FIELD_WIDTH))
        result[:, :] = self._field[:,:]
        if with_op:
            self._add_op_to_field(result, self._figure, self._figure_pos)
        return result

    def get_fullness(self):
        if self._field is None:
            return np.float32(0.0)
        return np.sum(np.sum(self._field)) / (self._field.shape[0]*self._field.shape[1])

    def get_op(self, figure, figure_pos):
        """
        return operation array
        :return: operation array
        """
        result = np.zeros((FIELD_FULL_HEIGHT, FIELD_WIDTH))
        self._add_op_to_field(result, figure, figure_pos)
        return result

    def get_op_info(self):
        op = OperationInfo(None, self._figure, self._figure_pos, Operation.NOP)
        return OperationInfo(
            self.get_op(op.figure, op.figure_pos)
            , op.figure
            , op.figure_pos
            , op.op
        )

    def get_ops(self, ops : List[OperationInfo]):
        result = [
            OperationInfo(
                self.get_op(op.figure, op.figure_pos)
                , op.figure
                , op.figure_pos
                , op.op
                )
            for op in ops
            ]
        return result

    def draw(self, game, screen, top, left, width, height, warn_map = None):
        """
        draw field and figure to the pygame screen
        :param game: game object
        :param screen: screen object
        :param top: top offset
        :param left: left offset
        :param width: width of draw area
        :param height: height of draw area
        :return:
        """
        field = self.get_field()
        w = field.shape[1]
        h = field.shape[0]
        for y in range(h):
            for x in range(w):
                c = 1.0 #warn_map[y,x] if warn_map is not None else 1.0
                clr = (0, 127, 255) \
                    if field[y,x] > 0 \
                    else (int(127*c)+32, int(127*c)+32, int(127*c)+32)
                game.draw.rect(screen, clr
                    , pygame.Rect(
                        int(width*x/w) + left
                        , height + top - int(height*(y+1)/h)
                        , int(width/w) + 1
                        , int(height/h) + 1
                    ))
        return

    pass #class Tetris


class PlayerTetrisGame:
    """
    pygame implementation of tetris game
    """

    K_LEFT = 0
    K_RIGHT = 1
    K_SPACE = 2
    K_DOWN = 3
    K_COUNT = 4

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((300, 800))
        self._done = False
        self._tetris = Tetris()
        self._clock = pygame.time.Clock()
        return

    def _check_key(self,keys, key_index, key_pressed):
        keys[key_index] = 1 if key_pressed else 0

    def _key_pressed(self, keys, oldkeys, key_index):
        return oldkeys[key_index] == 0 and keys[key_index] != 0

    def _gravity(self):
        if not self._tetris.gravity():
            self._tetris.add_figure()
            self._tetris.cut_lines()
            if not self._tetris.next_figure(np.random.randint(0, 100)):
                self._tetris.start()
                self._tetris.next_figure(np.random.randint(0,100))
        return

    def run(self):
        """
        run the tetris game of pygame
        :return:
        """

        self._tetris.start()
        self._tetris.next_figure(np.random.randint(0,100))
        t0 = pygame.time.get_ticks()
        oldkeys = [0] * self.K_COUNT

        while not self._done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True


            pressed = pygame.key.get_pressed()
            keys = [0] * self.K_COUNT
            self._check_key(keys, self.K_LEFT, pressed[pygame.K_LEFT])
            self._check_key(keys, self.K_RIGHT, pressed[pygame.K_RIGHT])
            self._check_key(keys, self.K_SPACE, pressed[pygame.K_SPACE])
            self._check_key(keys, self.K_DOWN, pressed[pygame.K_DOWN])
            if self._key_pressed(keys, oldkeys, self.K_LEFT): self._tetris.move_left()
            if self._key_pressed(keys, oldkeys, self.K_RIGHT): self._tetris.move_right()
            if self._key_pressed(keys, oldkeys, self.K_SPACE): self._tetris.rotate_figure_cw()
            if self._key_pressed(keys, oldkeys, self.K_DOWN):
                t1 = pygame.time.get_ticks()
                t0 = t1 - 1000
                pass

            oldkeys = keys

            t1 = pygame.time.get_ticks()
            if t1 - t0 >= 1000:
                self._gravity()
                t0 = (t1//1000) * 1000
                pass

            self._screen.fill((0, 0, 0))
            self._tetris.draw(pygame, self._screen, 10, 10, 280, 780)

            pygame.display.flip()
            self._clock.tick(60)
            pass
        pass

    pass #class PlayerTetrisGame


def test_tetris_game():
    game = PlayerTetrisGame()
    game.run()
    return

if __name__ == "__main__":
    test_tetris_game()