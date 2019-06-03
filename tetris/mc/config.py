import tensorflow as tf

VERBOSE = True
RL_SEARCH_COEF = 0.005
GAME_STATES_DB_SIZE_MAX = 1024*10
GAME_STATES_DB_SIZE_MAX2 = 1024*11
GAME_STATES_DB_SIZE_START_TRAIN = 1024*10
GAME_STATES_DB_SIZE_SAVE_SIZE = 1024*10
GAME_STATES_DB_STEP_SIZE = 100
MAX_FIGURE_SIZE = 4
FIELD_WIDTH = 8
FIELD_HEIGHT = 16
FIELD_FULL_HEIGHT = FIELD_HEIGHT + MAX_FIGURE_SIZE
STATES_TO_TRAIN_BATCH_SIZE=32
STATES_TO_TRAIN=STATES_TO_TRAIN_BATCH_SIZE
TRAIN_RO = 0.95
LEARN_RO = 0.75
END_GAME_REWARD_COEF=0.999
LINE_REWARD_COEF = 0.9
MODEL_WEIGHTS_PATH='/tmp/tetris/model.h5'
CNNMODEL_WEIGHTS_PATH = '/tmp/tetris/cnnpretrain/model{epoch:08d}.h5'
CNNMODEL_BEST_WEIGHTS_PATH = '/tmp/tetris/cnnpretrain/model.h5'
CNNMODEL_RUN_RESULTS_PATH = '/tmp/tetris/cnnpretrain/results.txt'
CNNMODEL_DATA_PATH = '/tmp/tetris/cnnpretrain/data/'
GAME_STATES_PATH = '/tmp/tetris/states'
TENSORBOARD_LOG_PATH = '/tmp/tetris/cnnpretrain/logs/'
MIN_REWARDED_STATES_RATE = 0.25
MAX_GAME_LENGTH = 100


NET_OPERATION_ITEM_SIZE = 8
NET_CNN_CONV2D_1_FILTERS = 32
NET_CNN_CONV2D_2_FILTERS = 64
NET_CNN_POOL_1 = 2
NET_CNN_POOL_2 = 2
NET1_FC_SIZE1 = (1024*3)//2
NET1_FC_SIZE2 = (1024*3)//2
NET1_FC_SIZE3 = 512
NET2_FC_SIZE1 = (1024*3)//2
NET2_FC_SIZE2 = 1024
NET1_FC_DROPOUT_VALUE1 = 0.2
NET1_FC_DROPOUT_VALUE2 = 0.1
NET2_FC_DROPOUT_VALUE = 0.2
NET_LAYER1_ACTIVATION=tf.nn.leaky_relu
NET_LAYER2_ACTIVATION=tf.nn.leaky_relu
