import os
from typing import AnyStr
from mc.config import *
import random
from nn.PlayedDB import PlayedGames, PlayedState


def get_train_files(path: AnyStr):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


class GameStat:
    def __init__(self, good=0, illegal=0, reward=0, operations=0):
        self.good = good
        self.illegal = illegal
        self.reward = reward
        self.operations = operations

    def add(self, other):
        self.good += other.good
        self.illegal += other.illegal
        self.reward += other.reward
        self.operations += other.operations

    def correct(self, other):
        self.good /= other
        self.illegal /= other
        self.reward /= other
        self.operations /= other

    def print(self):
        if VERBOSE:
            print("good = {0:.6} illegal = {1:.6} reward = {2:.6} operations = {3:.2}"
                  .format(self.good, self.illegal, self.reward, self.operations)
                  )


def state_stat(state: PlayedState):
    return GameStat(
        good=len(state.end_positions),
        illegal=len(state.illegal_positions),
        reward=sum([pos.get_reward() for pos in state.end_positions]) / (len(state.end_positions) + 1),
        operations=sum([
            len(pos.possible_ops) if pos.possible_ops else 0
            for pos in state.end_positions
        ]) / len(state.end_positions)
    )


def analyse_data(file: AnyStr):
    pg = PlayedGames(max_size=GAME_STATES_DB_SIZE_MAX)
    pg.load(file)
    stat = GameStat()
    for i in range(len(pg)):
        s = state_stat(pg.get_state(i))
        stat.add(s)
    stat.correct(len(pg))
    return stat


def analyse():
    files = get_train_files(CNNMODEL_DATA_PATH)
    stat = GameStat()
    i0 = 0
    for i, file in enumerate(files):
        stat.add(analyse_data(os.path.join(CNNMODEL_DATA_PATH,file)))
        i1 = (i/len(files)*100)
        if i0 < int(i1):
            print("{}%".format(i1))
            i0 = int(i1)
    stat.correct(len(files))
    stat.print()


def main():
    analyse()


if __name__=="__main__":
    main()