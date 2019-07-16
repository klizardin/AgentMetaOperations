from RLParking.settings import settings

import tensorflow as tf
import numpy as np


class NetData:
    def __init__(self, batch):
        self.x = batch
        self.y = None

    pass  # class NetData


class NetTrainData(NetData):
    def __init__(self, batch_x, batch_y):
        super(NetTrainData, self).__init__(batch_x)
        self.y = batch_y

    pass  # class NetTrainData


class OperationLayer(tf.layers.Layer):
    """
    layer to present operation in the net data
    """

    def __init__(self, items_count, item_sz, extra_sz, **kwargs):
        self._items_count = items_count
        self._item_sz = item_sz
        self._extra_sz = extra_sz
        self._expand_op = None
        self._op_base = None
        super(OperationLayer, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        a = np.zeros((self._items_count,), dtype=np.float32)
        a[:] = np.float32(0.5)
        e = np.zeros((self._items_count, self._items_count*self._item_sz), dtype=np.float32)
        for i in range(self._items_count):
            e[i, i*self._item_sz:(i+1)*self._item_sz] = np.float32(1.0)
        self._expand_op = tf.constant(e)
        self._op_base = tf.constant(a)
        super(OperationLayer, self).build(input_shape)
        return

    def call(self, inputs, op, **kwargs):
        """
        to implement layer operation
        :param inputs: input tensor
        :param op: operation numpy array
        :param kwargs: another arguments
        :return: result of operation
        """
        if self._extra_sz > 0:
            i1 = tf.slice(inputs, [0, 0], [-1, self._items_count * self._item_sz])
            i2 = tf.slice(inputs,
                          [0, self._items_count * self._item_sz],
                          [-1, self._extra_sz]
                          )
            op = tf.add(op, self._op_base)
            op1 = tf.matmul(op, self._expand_op)
            return tf.concat([tf.multiply(i1, op1), i2], 1)
        else:
            op = tf.add(op, self._op_base)
            op1 = tf.matmul(op, self._expand_op)
            return tf.multiply(inputs, op1)

    def compute_output_shape(self, input_shape):
        """
        to computer output shape from input shape
        :param input_shape:
        :return:
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self._items_count * self._item_sz + self._extra_sz
        return tf.TensorShape(shape)

    def get_config(self):
        """
        to serialize layer data
        :return: layer config
        """
        base_config = super(OperationLayer, self).get_config()
        base_config['items_count'] = self._items_count
        base_config['item_sz'] = self._item_sz
        base_config['extra_sz'] = self._extra_sz
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        create layer from the serialized data
        :param config: data to unserialize from
        :return:
        """
        return cls(**config)

    pass  # class OperationLayer


class RLModel(tf.keras.Model):
    """
    RL model class
    """
    def __init__(self):
        super(RLModel, self).__init__(name="parking_model")
        self._input_size = settings.NET_INPUT_SIZE
        self._op_count = settings.OPERATIONS_COUNT
        assert(settings.NET_OPERATION_EXTRA_SIZE >= 0)
        self._layers1 = list()
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE1))
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._fc_size = self._op_count * settings.NET_OPERATION_ITEM_SIZE + settings.NET_OPERATION_EXTRA_SIZE
        self._layers1.append(tf.layers.Dense(self._fc_size, activation=settings.NET_LAYER1_ACTIVATION))
        self._op_layer = OperationLayer(
            items_count=self._op_count,
            item_sz=settings.NET_OPERATION_ITEM_SIZE,
            extra_sz=settings.NET_OPERATION_EXTRA_SIZE
        )
        self._layers2 = list()
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE2, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE2, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(1))
        pass

    def call(self, inputs):
        x = tf.slice(inputs, [0, 0], [-1, self._input_size - self._op_count])
        op = tf.slice(inputs, [0, self._input_size - self._op_count], [-1, self._op_count])
        x = tf.reshape(x, [-1, self._input_size - self._op_count])
        for l in self._layers1:
            x = l(x)
        x = self._op_layer(x, op=op)
        for l in self._layers2:
            x = l(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = 1
        return tf.TensorShape(shape)

    @property
    def layers_p1(self):
        return self._layers1

    @property
    def layers_p2(self):
        return self._layers2

    pass  # RLModel
