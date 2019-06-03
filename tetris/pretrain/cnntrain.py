from mc.config import *
from nn.net import CNNPreTrainModel
from nn.PlayedDB import PlayedGames, PlayedState
from game.tetris import Tetris, Operation
from game.BotGame import TetrisPositionInfo
from typing import AnyStr
import numpy as np
import os
import random


def get_train_files(path: AnyStr):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def has_reward(state: PlayedState):
    return any([pt.has_reward() for pt in state.end_positions])


def get_state_cnn_value(field, illegal, pos:TetrisPositionInfo, dot_fill, tetris: Tetris):
    dot_fill = np.array([dot_fill,], dtype=np.float32)
    illegal = np.array([0.5 if illegal else -0.5,], dtype=np.float32)
    reward = np.array([
        0.5 if (pos.reward > 0.0) and (pos.reward < 1.5) else -0.5,
        0.5 if (pos.reward >= 1.5) and (pos.reward < 2.5) else -0.5,
        0.5 if (pos.reward >= 2.5) and (pos.reward < 3.5) else -0.5,
        0.5 if pos.reward >= 3.5 else -0.5
        ], dtype=np.float32
        )
    if pos.possible_ops is not None:
        ops = [
            0.5 if Operation.ROTATE_UCW in pos.possible_ops else -0.5,
            0.5 if Operation.ROTATE_CW in pos.possible_ops else -0.5,
            0.5 if Operation.MOVE_LEFT in pos.possible_ops else -0.5,
            0.5 if Operation.MOVE_RIGHT in pos.possible_ops else -0.5,
            0.5 if Operation.NOP in pos.possible_ops else -0.5,
        ]
    else:
        ops = [0.0, 0.0, 0.0, 0.0, 0.0]
    operations = np.array(ops,dtype=np.float32)
    return np.concatenate(((field - np.float32(0.5))*np.float32(0.25), illegal, operations, reward, dot_fill))


def fill_dots(y,x,val,dots):
    if dots[y,x] != 0:
        return
    dots[y, x] = val
    if y > 0:
        fill_dots(y - 1,x,val,dots)
    if y < dots.shape[0] - 1:
        fill_dots(y + 1, x, val, dots)
    if x > 0:
        fill_dots(y, x - 1, val, dots)
    if x < dots.shape[1] - 1:
        fill_dots(y, x + 1, val, dots)


def find_zero(dots):
    for y in range(dots.shape[0]):
        for x in range(dots.shape[1]):
            if dots[y,x] == 0:
                return (y,x)
    return False


def count_dots(field):
    dots_count = 2
    dots = np.zeros(field.shape,dtype=np.int32)
    dots[:,:] = field.astype(dtype=np.int32)
    fill_dots(field.shape[0] - 1, 0, dots_count, dots)
    pos = find_zero(dots)
    while pos:
        dots_count += 1
        fill_dots(pos[0], pos[1], dots_count, dots)
        pos = find_zero(dots)
    return dots_count


def get_train_data(train_state: PlayedState):
    tetris = Tetris()
    tetris.state = train_state.tetris_state

    positions = list(train_state.end_positions) + list(train_state.illegal_positions)
    pos_size = len(positions)
    ind_size = pos_size

    if pos_size > STATES_TO_TRAIN_BATCH_SIZE:
        pos_size = STATES_TO_TRAIN_BATCH_SIZE
    if ind_size < pos_size:
        ind_size = pos_size

    indexs = list(range(ind_size))
    indexs = sorted(indexs, key=lambda i: positions[i].has_reward())
    cnt = sum([1 if positions[i].has_reward() else 0 for i in indexs])
    noreward_indexs = indexs[cnt:]
    random.shuffle(noreward_indexs)
    indexs[cnt:] = noreward_indexs
    indexs = indexs[:pos_size]
    random.shuffle(indexs)

    x = list()
    y = list()
    field = tetris.get_field(False).flatten()
    for i in indexs:
        t1 = tetris.copy()
        pos = positions[i % len(positions)]
        illegal = not t1.set_figure_pos(pos.x, pos.y, pos.r)
        dot_fill = 0.0
        if illegal:
            t1.set_figure_illegal_pos(pos.x, pos.y, pos.r)
        elif pos.reward == 0:
            f1 = count_dots(t1.get_field(False))
            f2 = count_dots(t1.get_field(True))
            dot_fill = 0.5 if f2 > f1 else -0.5

        x.append(np.concatenate((field, t1.get_op_info().op_arr.flatten())))
        y.append(get_state_cnn_value(field, illegal, pos, dot_fill, t1))

    return np.stack(x), np.stack(y)


def get_random_indexes(pg:PlayedGames):
    indexes = list(range(len(pg)))
    indexes = sorted(indexes, key=lambda i: not has_reward(pg.get_state(i)))
    cnt = sum([1 if has_reward(pg.get_state(i)) else 0 for i in indexes])
    no_reward_indexes = indexes[cnt:]
    random.shuffle(no_reward_indexes)
    indexes[cnt:] = no_reward_indexes
    indexes = indexes[:max((cnt*2, 1024))]
    random.shuffle(indexes)
    return indexes


def generate_data_to_pretrain_cnn(file_name: AnyStr = None, path: AnyStr = None):
    assert((file_name is not None) or (path is not None))

    if file_name is not None:
        pg = PlayedGames(GAME_STATES_DB_SIZE_MAX//2)
        pg.load(file_name)
        pg.tight_objects()
        while True:
            indexes = get_random_indexes(pg)
            for j in indexes:
                state = pg.get_state(j)
                x,y = get_train_data(state)
                yield (x,y)

    if path is not None:
        files = get_train_files(path)
        file_indexes = list(range(len(files)))
        while True:
            random.shuffle(file_indexes)
            for fi in file_indexes:
                f = os.path.join(path,files[fi])
                pg = PlayedGames(GAME_STATES_DB_SIZE_MAX//2)
                pg.load(f)
                pg.tight_objects()
                indexes = get_random_indexes(pg)
                for j in indexes:
                    state = pg.get_state(j)
                    x,y = get_train_data(state)
                    yield (x,y)


def main():
    output_size = (
        FIELD_WIDTH*FIELD_FULL_HEIGHT # field to reconstruct
        + 1 # is legal operation
        + 5 # operations possible
        + 4 # does get any reward
        + 1 # dot fill
        )
    model = CNNPreTrainModel(width=FIELD_WIDTH, height=FIELD_FULL_HEIGHT, output_size=output_size)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
        , loss='mse'
        , metrics=['mse']
        )

    x = np.ones((STATES_TO_TRAIN_BATCH_SIZE, FIELD_WIDTH * FIELD_FULL_HEIGHT * 2), dtype=np.float32)
    y = np.ones((STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
    model.fit(
        x, y,
        epochs=1, batch_size=STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
        callbacks=None
    )

    if os.path.isfile(CNNMODEL_BEST_WEIGHTS_PATH):
        model.load_weights(CNNMODEL_BEST_WEIGHTS_PATH)
        print("model loaded {0}".format(CNNMODEL_BEST_WEIGHTS_PATH))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=25, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_PATH),
        tf.keras.callbacks.ModelCheckpoint(CNNMODEL_WEIGHTS_PATH,
            save_weights_only=True,
            verbose=1)
    ]
    model.fit_generator(
        generate_data_to_pretrain_cnn(path=CNNMODEL_DATA_PATH),
        steps_per_epoch=1024*50, epochs=5000,
        callbacks=callbacks,
        validation_data=generate_data_to_pretrain_cnn(file_name=GAME_STATES_PATH),
        validation_steps=1024*10
    )


def print_result(x,y,y1,width,height,output_size,f):
    bi = random.randint(0,x.shape[0]-1)
    f.write("\n\n")
    field1 = []
    for yi in range(height):
        field_line = ""
        for xi in range(width):
            field_line += "x" if x[bi,xi+yi*width] > 0 else " "
        field_line += "\n"
        field1.append(field_line)
    f.write("field :\n")
    f.writelines(field1)
    field2 = []
    for yi in range(height):
        field_line = ""
        for xi in range(width):
            field_line += "x" if x[bi,xi+yi*width + width*height] > 0 else " "
        field_line += "\n"
        field2.append(field_line)
    f.write("operation :\n")
    f.writelines(field2)
    field3 = []
    for yi in range(height):
        field_line = ""
        for xi in range(width):
            field_line += "x" if y1[bi,xi+yi*width] > 0 else " "
        field_line += "\n"
        field3.append(field_line)
    f.write("result field :\n")
    f.writelines(field3)
    l1 = "original :"
    l2 = "result   :"
    for yi in range(width*height, output_size):
        l1 += "{0:.3f} ".format(y[bi,yi])
        l2 += "{0:.3f} ".format(y1[bi,yi])
    l1 += "\n"
    l2 += "\n"
    f.write(l1)
    f.write(l2)
    f.flush()


def evaluate():
    print("Evaluating...")
    output_size = (
        FIELD_WIDTH*FIELD_FULL_HEIGHT # field to reconstruct
        + 1 # is legal operation
        + 5 # operations possible
        + 4 # does get any reward
        + 1 # dot fill
        )
    model = CNNPreTrainModel(width=FIELD_WIDTH, height=FIELD_FULL_HEIGHT, output_size=output_size)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
        , loss='mse'
        , metrics=['mse']
        )

    x = np.ones((STATES_TO_TRAIN_BATCH_SIZE, FIELD_WIDTH * FIELD_FULL_HEIGHT * 2), dtype=np.float32)
    y = np.ones((STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
    model.fit(
        x, y,
        epochs=1, batch_size=STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
        callbacks=None
    )

    if os.path.isfile(CNNMODEL_BEST_WEIGHTS_PATH):
        model.load_weights(CNNMODEL_BEST_WEIGHTS_PATH)
        print("model loaded")

    with open(CNNMODEL_RUN_RESULTS_PATH,"wt") as f:
        gen = generate_data_to_pretrain_cnn(file_name=GAME_STATES_PATH)
        _ = next(gen)
        for i in range(1024*4):
            (x,y) = next(gen)
            y1 = model.predict(x=x,batch_size=x.shape[0])
            print_result(x,y,y1,FIELD_WIDTH,FIELD_FULL_HEIGHT,output_size,f)
        del gen

    res = model.evaluate_generator(
        generator=generate_data_to_pretrain_cnn(file_name=GAME_STATES_PATH),
        steps=1024*4
        )
    print("evaluated={0}".format(res))

    print("Done")


if __name__ == "__main__":
    main()
    #evaluate()
