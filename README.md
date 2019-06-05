# Neural net architecture for RL algorithm with possibility using of several operations in one moment

One of the classical neural net architecture for using in RL algorithm looks like:

![classic neural net architecture for RL algorithm](https://github.com/klizardin/AgentMetaOperations/blob/master/img/net.svg.png)

Where: inputs – are inputs of the neural net; FC – fully connected hidden layers (or some times CNN + FC layers); outputs – network outputs, offen it is softmax layer to get probabilities of possible operations. 

The disadvantage of this architecture is that it is difficult to implement the choice of several simultaneous actions at once.

To solve this problem, an architecture with a mask layer is proposed. The proposed architecture is as follows:

![proposed neural architecture for RL algorithm](https://github.com/klizardin/AgentMetaOperations/blob/master/img/net_with_mask_layer.svg.png)

This architecture is fully consistent with the classical architecture, but also includes an action mask layer. The only way out of this architecture is the value of the action value (a group of simultaneous actions). The action mask layer can be implemented in accordance with the pseudocode below:

```
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
        self._expand_op = np.zeros((self.items, self.items*self.item_size), \
		dtype=np.float32)
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
```

And the use of this code demonstrates the following code snippet:

```
items = 5
item_size = 10
extra_size = 20
l = Layer(items=items, item_size=item_size, extra_size=extra_size)
l.build()
inputs = np.random.rand(items*item_size+extra_size)
ops = np.random.randint(0, 2, (items,), dtype="int")
ops = ops.astype(dtype=np.float32) - np.float32(0.5)
result = l.call(inputs,ops)
```

From the layer code it is clear that for each action a neural network learns to form some representation of the action as a series of weights. And these views either go to the network after the mask layer or not. Depending on the operation, these weights can be performed with the task of some operation on the whole group of action weights (not only multiplication by [0,1]). In this way, a task is formed to calculate the value of a group of actions performed by the network. (In the classic case, the softmax layer calculated the value of all actions, in the proposed architecture, the neural network calculates the value of a group of selected actions.)

(For a definition of action value, see, for example, R.S. Sutton, E.G. Barto Reinforcement learning.)

## Examples of the use of the proposed architecture

### Tetris game

The idea of using this architecture for playing Tetris is as follows. At the inputs, we submit an image of a glass game Tetris (one pixel one square). We group individual actions into action groups. Evaluation of a single action for a neural network is a mask of the final position of a figure in a glass. The shape is set by its squares in the action mask in the action mask layer in the neural network. To select a group of actions, we choose the maximum assessment of the action (exit) from the list of all final positions of the current figure.

![tetris game](https://github.com/klizardin/AgentMetaOperations/blob/master/img/tetris_1.png)

Picture. A field (blue cells) and a falling figure (light gray cells) are shown. The final position of the figure is all possible positions of which the figure cannot move according to the rules of the game (not shown).

### Agent modeling the movement of the vehicle

In this case, each action of acceleration (several speeds of acceleration), deceleration (several possible speeds during deceleration), as well as several degrees of rotation, were simulated as elementary actions. It is clear that a turn action and one of the accelerations action can be involved at the same time, or only one turn action or one acceleration action. In this case, the architecture allows you to set several elementary actions at the same time to form a complex action.

![parking app screen](https://github.com/klizardin/AgentMetaOperations/blob/master/img/parking_1.png)

Picture. In addition to the field itself, for performing actions by a car model (on which the parking target is indicated by red-green cross), the inputs of the neural network (below) and action evaluation values for all possible actions in a given state of the model are also displayed.

## Other possible uses of architecture

Similarly, using Tetris architecture in a game can be used for other games, where the field can be given by rows of figures and can be taken several actions at the same time (for example, moving around the playing field).
In robotics, this architecture can play the role of a meta-network coordinating individual structural elements into a common ensemble.

Also, this architecture allows you to use transfer learning to pre-train the CNN part, and vice versa at the beginning to train the RL part of the neural network, and then train the CNN part on the already trained RL network on model data. In the example in the programming of the game tetris transfer learning was applied with training at the beginning of the CNN part and FC part to the action mask layer (that is transferred to the resulting network). In the task of parking, I will also plan to apply the CNN training of the part after the training of the RL part (ie, the “cherry” first).


P.S. (Looking for a job)
