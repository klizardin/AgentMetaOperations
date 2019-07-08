from RLParking.db import TrainState, TrainStates
import RLParking.settings as settings

import matplotlib.pyplot as plt


def plot_state(ts: TrainState):
    p1 = ts.state.pos
    p2 = ts.next_state.pos
    x = [p1.x, p2.x]
    y = [p1.y, p2.y]
    plt.plot(x,y, color="red" if ts.has_values() else "black")


def main():
    train_db = TrainStates(max_length=settings.GAME_STATES_IN_DB_SIZE_MAX,
                           reward_ratio=settings.GAME_STATES_IN_DB_REWARD_RATIO)
    for item in train_db.get_all_items():
        plot_state(item)
    plt.show()


if __name__ == "__main__":
    main()
