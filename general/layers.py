
# lib
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import (
    InputSpec,
    Layer,
    Dense,
    LocallyConnected1D,
    Convolution1D,
    TimeDistributed,
    Reshape,
    Flatten
)

from keras import (
    activations,
    constraints,
    initializers,
    regularizers,
    constraints,
)

from keras.engine.topology import InputSpec
from keras.layers.merge import _Merge

# self
from general.binary_ops import binarize


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}


class BinaryDense(Dense):
    """ Binarized Dense layer
    References:
    "BinaryNet: Training Deep Neural Networks with Weights
     and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    """

    def __init__(self, units, H=1., kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, **kwargs):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

        super(BinaryDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            # print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            # print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Intertwine(_Merge):
    """Layer that intertwines a list of inputs.

    It takes as input a list of tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the intertwination of all inputs

    esc

        Intertwine()([[a,b,c],[d,e,f]) := [a,d,b,e,c,f]

    # Arguments
        axis: Axis along whic   h to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Intertwine, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of at least 2 inputs')
        if all([shape is None for shape in input_shape]):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('A `Intertwine` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shape))

    def _merge_function(self, inputs):

        shape = list(K.int_shape(inputs[0]))
        shape[self.axis] = 2*shape[self.axis]
        shape[0] = -1

        return tf.reshape(tf.stack(inputs, axis=self.axis), shape)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Intertwine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ElementWise:

    def __init__(self,
                 units,
                 activation=None,
                 share_element_weights=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name=None):
        """
        An element-wise transform of several layers. For example,
        given two inputs:

            [a,b,c] and [e,f,g]

        apply a neural network that transforms [a,e], [b,f], and [c,g] seperatly
        without input from the other inputs. The depth (number of layers)
        of this function is configurable.

        :param units: the units of the element wise function. Can be a single
                      scalar or a list of scalars. With a list of scalars, multiple
                      layers will be applied along the extracted element-wise input.

        :param activation: the activation function of the layers. Can be a single element
                           or a list of elements. If a list, it must be the same length
                           as the given units list

        :param share_element_weights: whether or not to share the weights for the element-wise
                                      function.

        see keras documentation for additional details about other
        initializer properties.
        """

        super(ElementWise, self).__init__()

        if type(units) not in [list, tuple]:
            units = [units]

        self.units = units

        if type(activation) not in [list, tuple]:
            activation = [activation]*len(self.units)
        else:
            if len(activation) != len(self.units):
                raise ValueError('a list of activation functions must be '
                                 'the same length as the list of input units')

        self.activation = [activations.get(e) for e in activation]
        self.share_element_weights = share_element_weights
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = False

        self.name = name

    def __call__(self, inputs):
        """
        Apply an element-wise function as built by the
        initializer in this class

        :param inputs: a list of keras layers. len(inputs) >= 2. These
                       tensors must each be of shape

                            (batch_size, timesteps, 1)

                       each element at each input will all be inputs
                       into their own smaller neural network as described
                       by the construction of this object.
        :return: the result of the element-wise function. The result
                 will be of shape

                           (batch_size, timesteps, units.last
        """

        if len(inputs) < 2:
            raise ValueError('Element wise functions require at least 2 inputs')

        initial_connection_size = len(inputs)

        """
        Connect the input pairs by interleaving them among one another.
        1D Convolutions (or LocallyConnected) can then be used to 
        make local functions of the individual inputs.
        
        """
        intertwine = Intertwine(name=(self.name + '_intertwine') if self.name is not None else None)([
            *inputs
        ])

        reshape = Reshape(
           (-1, 1), name=(self.name + '_reshape') if self.name is not None else None
        )(intertwine)

        """
        extract the individual elements out into their own layer 
        of shape
        
            (batch_size, timesteps, units.first)
        """

        if self.share_element_weights:
            Connection = Convolution1D
        else:
            Connection = LocallyConnected1D

        extraction = Connection(self.units[0],
                                kernel_size=initial_connection_size,
                                strides=initial_connection_size,
                                padding='valid',
                                activation=self.activation[0],
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                activity_regularizer=self.activity_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint,
                                name=(self.name + '_connection') if self.name is not None else None)(reshape)


        """
        apply additional transforms along all units given 
        of shape  
        
            (batch_size, timesteps, units.first)
        """
        num = 0
        if len(self.units) > 1:

            for units, activation in zip(self.units[1:], self.activation[1:]):

                num += 1
                dense = Dense(
                      units,
                      activation=activation,
                      use_bias=self.use_bias,
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      bias_regularizer=self.bias_regularizer,
                      activity_regularizer=self.activity_regularizer,
                      kernel_constraint=self.kernel_constraint,
                      bias_constraint=self.bias_constraint,
                      name=(self.name + '_dense_' + str(num)) if self.name is not None else None
                )

                if not self.share_element_weights:
                    Connection = TimeDistributed(
                        dense, name=(self.name + '_time_dist_' + str(num)) if self.name is not None else None
                    )
                else:
                    Connection = dense

                extraction = Connection(extraction)

        return extraction










