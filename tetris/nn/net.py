import numpy as np
import tensorflow as tf
from mc.config import *


class OperationLayer(tf.layers.Layer):
    """
    layer to present operation in the net data
    """

    def __init__(self, width, height, item_sz, extra_sz, **kwargs):
        self._width = width
        self._height = height
        self._item_sz = item_sz
        self._extra_sz = extra_sz
        super(OperationLayer, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        e = np.zeros((self._width*self._height, self._width*self._height*self._item_sz), dtype=np.float32)
        for i in range(self._width*self._height):
            e[i,i*self._item_sz:(i+1)*self._item_sz] = np.float32(1.0)
        self._expand_op = tf.constant(e)
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
        i1 = tf.slice(inputs, [0, 0], [-1, self._height * self._width*self._item_sz])
        i2 = tf.slice(inputs,
            [0, self._height * self._width * self._item_sz]
            , [-1, self._extra_sz]
            )
        op1 = tf.matmul(op, self._expand_op)
        return tf.concat([tf.multiply(i1,op1),i2],1)

    def compute_output_shape(self, input_shape):
        """
        to computer output shape from input shape
        :param input_shape:
        :return:
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self._height*self._width*self._item_sz + self._extra_sz
        return tf.TensorShape(shape)

    def get_config(self):
        """
        to serialize layer data
        :return: layer config
        """
        base_config = super(OperationLayer, self).get_config()
        base_config['width'] = self._width
        base_config['height'] = self._height
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

    pass #class OperationLayer


class CNNPreTrainModel(tf.keras.Model):
    """
    RL model class
    """
    def __init__(self, *, width, height, output_size):
        super(CNNPreTrainModel, self).__init__(name="cnn_tetris_model")
        self._width = int(width)
        self._height = int(height)
        self.OPERATION_EXTRA_SIZE = NET2_FC_SIZE1 - self._width*self._height*NET_OPERATION_ITEM_SIZE
        assert(self.OPERATION_EXTRA_SIZE>=128)
        self._layers_cnn = list()
        self._layers1 = list()
        self._layers2 = list()
        self._layers_cnn.append(tf.keras.layers.Conv2D(
            filters=NET_CNN_CONV2D_1_FILTERS
            , kernel_size=(5,5)
            , padding="same"
            , activation=tf.nn.leaky_relu
            #, input_shape=(self._width * self._height * 2,)
            ))
        self._layers_cnn.append(tf.layers.MaxPooling2D(
            pool_size=(NET_CNN_POOL_1,NET_CNN_POOL_1)
            , strides=(NET_CNN_POOL_1,NET_CNN_POOL_1)
            ))
        self._layers_cnn.append(tf.keras.layers.Conv2D(
            filters=NET_CNN_CONV2D_2_FILTERS
            , kernel_size=(5,5)
            , padding="same"
            , activation=tf.nn.leaky_relu
        ))
        self._layers_cnn.append(tf.layers.MaxPooling2D(
            pool_size=(NET_CNN_POOL_2,NET_CNN_POOL_2)
            , strides=(NET_CNN_POOL_2,NET_CNN_POOL_2)
        ))
        self._layers1.append(tf.layers.Dense(NET2_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        #self._layers1.append(tf.layers.Dropout(NET_FC_DROPOUT_VALUE))
        self._layers1.append(tf.layers.Dense(NET2_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dropout(NET2_FC_DROPOUT_VALUE))
        self._layers1.append(tf.layers.Dense(NET2_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        self._fc_size = self._width * self._height * NET_OPERATION_ITEM_SIZE + self.OPERATION_EXTRA_SIZE
        self._layers1.append(tf.layers.Dense(self._fc_size, activation=NET_LAYER1_ACTIVATION))
        self._op_layer = OperationLayer(
            width=self._width,
            height=self._height,
            item_sz=NET_OPERATION_ITEM_SIZE,
            extra_sz=self.OPERATION_EXTRA_SIZE,
        )
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(NET2_FC_DROPOUT_VALUE))
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(NET2_FC_DROPOUT_VALUE))
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET2_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(output_size))
        pass

    @property
    def cnn_layers(self):
        return self._layers_cnn

    @property
    def layers_p1(self):
        return self._layers1

    def call(self, inputs):
        x = tf.slice(inputs, [0, 0], [-1, self._width*self._height])
        op = tf.slice(inputs, [0, self._width*self._height], [-1, self._width*self._height])
        x = tf.reshape(x, [-1,self._height,self._width, 1])
        for l in self._layers_cnn:
            x = l(x)
        x = tf.reshape(x, [-1,
                (self._height // (NET_CNN_POOL_1*NET_CNN_POOL_2))
                * (self._width // (NET_CNN_POOL_1*NET_CNN_POOL_2))
                * NET_CNN_CONV2D_2_FILTERS
                ])
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

    pass # class CNNPreTrainModel


class RLModel(tf.keras.Model):
    """
    RL model class
    """
    def __init__(self, *, width, height):
        super(RLModel, self).__init__(name="tetris_model")
        self._width = int(width)
        self._height = int(height)
        self.OPERATION_EXTRA_SIZE = NET1_FC_SIZE1 - self._width*self._height*NET_OPERATION_ITEM_SIZE
        assert(self.OPERATION_EXTRA_SIZE>=128)
        self._layers_cnn = list()
        self._layers1 = list()
        self._layers2 = list()
        self._layers_cnn.append(tf.keras.layers.Conv2D(
            filters=NET_CNN_CONV2D_1_FILTERS
            , kernel_size=(5,5)
            , padding="same"
            , activation=tf.nn.leaky_relu))
        self._layers_cnn.append(tf.layers.MaxPooling2D(
            pool_size=(NET_CNN_POOL_1,NET_CNN_POOL_1)
            , strides=(NET_CNN_POOL_1,NET_CNN_POOL_1)))
        self._layers_cnn.append(tf.keras.layers.Conv2D(
            filters=NET_CNN_CONV2D_2_FILTERS
            , kernel_size=(5,5)
            , padding="same"
            , activation=tf.nn.leaky_relu))
        self._layers_cnn.append(tf.layers.MaxPooling2D(
            pool_size=(NET_CNN_POOL_2,NET_CNN_POOL_2)
            , strides=(NET_CNN_POOL_2,NET_CNN_POOL_2)))
        self._layers1.append(tf.layers.Dense(NET1_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        #self._layers1.append(tf.layers.Dropout(NET_FC_DROPOUT_VALUE))
        self._layers1.append(tf.layers.Dense(NET1_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dropout(NET1_FC_DROPOUT_VALUE1))
        self._layers1.append(tf.layers.Dense(NET1_FC_SIZE1, activation=NET_LAYER1_ACTIVATION))
        self._fc_size = self._width * self._height * NET_OPERATION_ITEM_SIZE + self.OPERATION_EXTRA_SIZE
        self._layers1.append(tf.layers.Dense(self._fc_size, activation=NET_LAYER1_ACTIVATION))
        self._op_layer = OperationLayer(
            width=self._width,
            height=self._height,
            item_sz=NET_OPERATION_ITEM_SIZE,
            extra_sz=self.OPERATION_EXTRA_SIZE)
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE2, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE3, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE3, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE3, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(NET1_FC_SIZE3, activation=NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(1))
        pass

    def call(self, inputs):
        x = tf.slice(inputs, [0, 0], [-1, self._width*self._height])
        op = tf.slice(inputs, [0, self._width*self._height], [-1, self._width*self._height])
        x = tf.reshape(x, [-1,self._height,self._width, 1])
        for l in self._layers_cnn:
            x = l(x)
        x = tf.reshape(x, [-1,
                (self._height // (NET_CNN_POOL_1*NET_CNN_POOL_2))
                * (self._width // (NET_CNN_POOL_1*NET_CNN_POOL_2))
                * NET_CNN_CONV2D_2_FILTERS
                ])
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
    def cnn_layers(self):
        return self._layers_cnn

    @property
    def layers_p1(self):
        return self._layers1

    def print_weights(self):
        for l in self._layers_cnn:
            print(l.get_weights())
        for l in self._layers1:
            print(l.get_weights())
        for l in self._layers2:
            print(l.get_weights())

    def copy_weights_from(self,cnn_model: CNNPreTrainModel):
        assert(len(self._layers_cnn) == len(cnn_model.cnn_layers))
        assert(len(self._layers1) == len(cnn_model.layers_p1))
        for l_to,l_from in zip(self._layers_cnn,cnn_model.cnn_layers):
            l_to.set_weights(l_from.get_weights())
        for l_to,l_from in zip(self._layers1, cnn_model.layers_p1):
            l_to.set_weights(l_from.get_weights())

    def freeze_cnn_weights(self):
        for l in self._layers_cnn:
            l.trainable = False
        for l in self._layers1:
            l.trainable = False

    pass #RLModel


"""
import numpy as np

class Layer:
    def __init__(self, items, item_size, extra_size):
        assert(items > 0)
        assert(item_size > 0)
        assert(extra_size >= 0)
        self.items = items
        self.item_size = item_size
        self.extra_size = extra_size

    def build(self):
        self._expand_op = np.zeros((self.items, self.items*self.item_size), dtype=np.float32)
        for i in range(self.items):
            self._expand_op[i,i*self.item_size:(i+1)*self.item_size] = np.float32(1.0)

    def call(self, inputs, ops):
        op_mask_part = inputs[:self.items*self.item_size]
        if self.extra_size > 0:
            ext_part = inputs[self.items*self.item_size:]
        else:
            ext_part = None

        # if ops in [-0.5, 0.5] or [-0.5 .. 0.5]:
        ops1 = np.add(ops, np.float(0.5)) # optional

        extended_op = np.matmul(ops1, self._expand_op)

        if self.extra_size > 0:
            return np.concatenate((np.multiply(op_mask_part, extended_op), ext_part))
        else:
            return np.multiply(op_mask_part,extended_op)

def main():
    items = 5
    item_size = 10
    extra_size = 0
    l = Layer(items=items, item_size=item_size, extra_size=extra_size)
    l.build()
    inputs = np.random.rand(items*item_size+extra_size)
    ops = np.random.randint(0, 2, (items,), dtype="int")
    ops = ops.astype(dtype=np.float32) - np.float32(0.5)
    result = l.call(inputs,ops)
    print("{}".format(inputs))
    print("{}".format(ops))
    print("{}".format(result))

"""