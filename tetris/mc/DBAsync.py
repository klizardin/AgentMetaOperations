import numpy as np

from game.tetris import FieldState, Tetris, TetrisState, Position, OperationInfo
from mc.config import *
from mc.request import *
from nn.PlayedDB import PlayedGames, PlayedState
from game.BotGame import get_tetris_positions, get_tetris_end_positions
import random
import copy
import uuid
import os
import sys

def calc_reward(start_state: bool, finit_state: bool, original_reward, line : np.float32, fullness : np.float32):
    if start_state:
        return 1.0
    if finit_state:
        return -1.0
    if original_reward != 0:
        return original_reward / MAX_FIGURE_SIZE
    d = fullness - line
    return (d + 0.5
            if d + 0.1 >= 0.0
            else d - 0.5
            ) * 0.075



def propagate_value(value):
    v = np.float32(TRAIN_RO) * value
    return np.clip(v, np.float32(-0.5), np.float32(0.5))


def calc_value(start_state: bool, finit_state: bool, reward, best_next_value):
    reward = np.float32(reward)*np.float32(0.5)
    if finit_state or start_state:
        value = np.full(best_next_value.shape, reward, dtype=np.float32)
    else:
        value = propagate_value(best_next_value) + reward
    return np.clip(value, np.float32(-0.5), np.float32(0.5))


class DBAsync:
    def __init__(self, *, db : PlayedGames):
        self._db = db

    @staticmethod
    def _get_end_positions_values(train_state: PlayedState, tetris : Tetris):
        tetris.state = train_state.tetris_state
        z = np.zeros((1,), dtype=np.float32)
        for ep in train_state.end_positions:
            ep.value = None
            t1 = tetris.copy()
            assert(t1.set_figure_pos(ep.x, ep.y, ep.r))
            assert(not t1.gravity())
            assert(t1.add_figure())
            field_states = list()
            for fi in range(tetris.figures_count):
                t2 = t1.copy()
                if ep.finit_state or ep.start_state:
                    break
                if not t2.next_figure(fi):
                    ep.finit_state = True
                    break
                if t2.is_empty_field():
                    ep.start_state = True
                    break
                end_positions = get_tetris_end_positions(t2)
                ops_info = list()
                for ep1 in end_positions:
                    t3 = t2.copy()
                    assert (t3.set_figure_pos(ep1.x, ep1.y, ep1.r))
                    assert (not t3.gravity())
                    ops_info.append(t3.get_op_info())
                field_states.append(
                    FieldState(
                        field=t2.get_field(False), op=None, ops=ops_info
                    )
                )
                pass

            t1 = tetris.copy()
            t1.set_figure_pos(ep.x, ep.y, ep.r)
            fullness = t1.get_fullness()
            line = t1.figure_line / FIELD_FULL_HEIGHT

            if ep.finit_state or ep.start_state:
                ep.value = propagate_value(
                    calc_value(ep.start_state, ep.finit_state,
                        calc_reward(ep.start_state, ep.finit_state, ep.reward, line , fullness),
                        z)
                    )
            else:
                request = yield Request(
                    RequestType.GET_BEST_OPERATION_BATCH,
                    field_states=field_states
                )
                avg = np.array([x.get_best_op().value[0] for x in request.field_states],dtype=np.float32)
                avg = np.sum(avg)
                avg /= tetris.figures_count
                avg = np.array([avg], dtype=np.float32)
                ep.value = propagate_value(
                    calc_value(False, False,
                        calc_reward(False, False, ep.reward, line, fullness),
                        avg)
                    )

    @staticmethod
    def _get_field_values(train_state: PlayedState):
        tetris = Tetris()
        tetris.state = train_state.tetris_state
        field = tetris.get_field(False)
        counts = np.ones(field.shape, dtype=np.float32)
        for pt in train_state.end_positions:
            fw = train_state.tetris_state.figure.shape[1]
            fh = train_state.tetris_state.figure.shape[0]
            pos_tl = Position(x=pt.x - fw // 2, y=pt.y - fh // 2)
            #assert(field[pt.x, pt.y] == np.float32(0.0))
            if pt.value is None:
                field[pos_tl.y, pos_tl.x] = np.float32(-1.0)
                counts[pos_tl.y, pos_tl.x] = np.float32(1.0)
            else:
                field[pos_tl.y, pos_tl.x] += (pt.value / pt.count)[0]
                counts[pos_tl.y, pos_tl.x] += np.float32(1.0)
        field = np.divide(field, counts)
        res = list()
        for y in range(field.shape[0]):
            res.append("".join(["{:0.6f} ".format(field[y,x]) for x in range(field.shape[1])]))
        return field, res

    @staticmethod
    def _get_states_to_train(train_state: PlayedState, tetris : Tetris):
        tetris.state = train_state.tetris_state
        positions = [pt for pt in train_state.end_positions if pt.value is not None]
        assert(positions)

        field_states = list()
        for pos in positions:
            t1 = tetris.copy()
            assert(t1.set_figure_pos(pos.x, pos.y, pos.r))
            fs = FieldState(field=t1.get_field(False), op=t1.get_op_info(), value=np.copy(pos.value))
            field_states.append(fs)

        return field_states

    def process(self):
        db_size = 0
        while len(self._db) < GAME_STATES_DB_SIZE_START_TRAIN:
            _ = yield Request(RequestType.NOP)
            if len(self._db) >= db_size + GAME_STATES_DB_STEP_SIZE:
                self._db.print_size()
                db_size = (len(self._db) // GAME_STATES_DB_STEP_SIZE) * GAME_STATES_DB_STEP_SIZE

        tetris = Tetris()
        while True:
            yield from self._process_main(tetris)

            # output db size
            if len(self._db) >= db_size + GAME_STATES_DB_STEP_SIZE:
                self._db.print_size()
                db_size = (len(self._db) // GAME_STATES_DB_STEP_SIZE) * GAME_STATES_DB_STEP_SIZE
            # check states db size
            if len(self._db) > GAME_STATES_DB_SIZE_MAX2:
                self._db.tight_objects()
                self._db.save(GAME_STATES_PATH)
                db_size = len(self._db)
        return

    def _process_main(self, tetris):
        # get state to train
        train_field_states = list()
        while len(train_field_states) < STATES_TO_TRAIN:
            db_state = copy.deepcopy(self._db.get_rand_state())
            # get field states of end points to work with
            yield from DBAsync._get_end_positions_values(db_state, tetris)
            # get states to train
            train_field_states.extend(DBAsync._get_states_to_train(db_state, tetris))

        #random.shuffle(train_field_states)
        train_field_states = train_field_states[:STATES_TO_TRAIN]
        # train
        request = yield Request(
            RequestType.TRAIN,
            field_states=train_field_states
        )

    pass # class DBAsync


class AsyncCNNTrainDataCreater:
    def __init__(self, *, db : PlayedGames):
        self._db = db

    def process(self):
        data_created_size = 0
        db_size = 0
        while True:
            data_created_size, db_size = yield from self._process_main(data_created_size, db_size)
            if len(self._db) >= db_size + GAME_STATES_DB_STEP_SIZE:
                self._db.print_size()
                db_size = (len(self._db) // GAME_STATES_DB_STEP_SIZE) * GAME_STATES_DB_STEP_SIZE
                if VERBOSE:
                    print("created db size = {}".format(data_created_size))
        return

    def _process_main(self, data_created_size, db_size):
        if len(self._db) < GAME_STATES_DB_SIZE_SAVE_SIZE:
            _ = yield Request(RequestType.NOP)
        else:
            unique_filename = os.path.join(CNNMODEL_DATA_PATH, str(uuid.uuid4()))
            data_created_size += len(self._db)
            self._db.save(unique_filename)
            self._db.clear()
            db_size = 0
        return data_created_size, db_size

    pass # class AsyncCNNTrainDataCreater