from RLParking.request import RequestType, Request
from RLParking.db import TrainState, TrainStates
from env.vehicle import Vehicle, StaticEnv
import RLParking.settings as settings

from typing import List
import numpy as np


class AsyncTrain:
    def __init__(self, db: TrainStates, env: StaticEnv):
        self._db = db
        self._env = env

    def process(self):
        try:
            db_size = 0
            while len(self._db) < settings.GAME_STATES_IN_DB_TO_START_TRAIN:
                _ = yield Request(RequestType.NOP)
                if len(self._db) >= db_size + settings.GAME_STATES_IN_DB_STEP_SIZE:
                    self._db.print_size()
                    db_size = (len(self._db) // settings.GAME_STATES_IN_DB_STEP_SIZE) * settings.GAME_STATES_IN_DB_STEP_SIZE

            vehicle = Vehicle("TrainVehicle")
            while True:
                yield from self._process_main(vehicle)

                # output db size
                if len(self._db) >= db_size + settings.GAME_STATES_IN_DB_STEP_SIZE:
                    self._db.print_size()
                    db_size = (len(self._db) // settings.GAME_STATES_IN_DB_STEP_SIZE) * settings.GAME_STATES_IN_DB_STEP_SIZE
                # check states db size
                if len(self._db) > settings.GAME_STATES_IN_DB_SIZE_MAX2:
                    self._db.reduce_size()
                    self._db.save(settings.PATHES.get_train_state())
                    db_size = len(self._db)
        except StopIteration as e:
            return

    def _prepare_inputs(self, states: List[TrainState], vehicle: Vehicle):
        inputs = list()
        ops = list()
        for train_state in states:
            vehicle.state = train_state.state
            state_inputs = vehicle.get_input_coefs(self._env)
            state_inputs = np.array(state_inputs, dtype=np.float32)
            sensor_inputs = vehicle.get_sensor_inputs()
            state_inputs = np.concatenate((state_inputs.flatten(), sensor_inputs))
            inputs.append(state_inputs)
            #ops.append(train_state.state.get_last_commands())
            ops.append(vehicle.get_last_operation_info())
        return inputs, ops

    def _prepare_values(self, states: List[TrainState], vehicle: Vehicle):
        values = np.zeros((len(states),),dtype=np.float32)
        for i,train_state in enumerate(states):
            if train_state.final_state:
                values[i] = train_state.reward
            else:
                vehicle.state = train_state.next_state
                ops = vehicle.get_next_available_ops()
                ops_inputs = vehicle.model_ops(ops, settings.VEHICLE_STEP_DURATION);
                inputs = vehicle.get_input_coefs(self._env)
                inputs = np.array(inputs, dtype=np.float32)
                sensor_inputs = vehicle.get_sensor_inputs()
                inputs = np.concatenate((inputs.flatten(), sensor_inputs))
                request = yield Request(
                    type = RequestType.GET_BEST_OPERATION,
                    inputs = inputs,
                    ops = ops_inputs
                )
                values[i] = (
                    request.get_best_operation_value()*settings.VALUES_TO_REWARD_COEF*settings.RL_RO_COEF
                    + train_state.reward
                    )
        return np.clip(values * settings.REWARD_TO_VALUES_COEF, np.float32(-0.5), np.float32(0.5))

    def _prepare_inputs_values2(self, states: List[TrainState], vehicle: Vehicle):
        inputs = list()
        values = list()
        ops = list()
        for i, train_state in enumerate(states):
            if train_state.has_values():
                vehicle.state = train_state.state
                state_inputs = vehicle.get_input_coefs(self._env)
                state_inputs = np.array(state_inputs, dtype=np.float32)
                sensor_inputs = vehicle.get_sensor_inputs()
                state_inputs = np.concatenate((state_inputs.flatten(), sensor_inputs))
                for op in train_state.ops:
                    inputs.append(state_inputs)
                    ops.append(op)
                values.append(train_state.values)
        if values:
            values = np.stack(values).flatten()
            values = np.clip(values * settings.REWARD_TO_VALUES_COEF, np.float32(-0.5), np.float32(0.5))
        return inputs, ops, values

    def _process_main(self, vehicle: Vehicle):
        states = self._db.get_items(count=settings.STATES_TO_TRAIN)
        inputs, ops = self._prepare_inputs(states, vehicle)
        values = yield from self._prepare_values(states, vehicle)
        _ = yield Request(
            type=RequestType.RL_TRAIN,
            inputs=inputs,
            ops=ops,
            values=values
        )
        inputs, ops, values = self._prepare_inputs_values2(states, vehicle)
        if inputs:
            _ = yield Request(
                type=RequestType.RL_TRAIN,
                inputs=inputs,
                ops=ops,
                values=values
            )

    pass # class AsyncTrain