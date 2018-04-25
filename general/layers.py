
# lib
import tensorflow as tf

from keras.layers.merge import _Merge
import keras.backend as K


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

    # def compute_mask(self, inputs, mask=None):
    #     if mask is None:
    #         return None
    #     if not isinstance(mask, list):
    #         raise ValueError('`mask` should be a list.')
    #     if not isinstance(inputs, list):
    #         raise ValueError('`inputs` should be a list.')
    #     if len(mask) != len(inputs):
    #         raise ValueError('The lists `inputs` and `mask` '
    #                          'should have the same length.')
    #     if all([m is None for m in mask]):
    #         return None
    #     # Make a list of masks while making sure
    #     # the dimensionality of each mask
    #     # is the same as the corresponding input.
    #     masks = []
    #     for input_i, mask_i in zip(inputs, mask):
    #         if mask_i is None:
    #             # Input is unmasked. Append all 1s to masks,
    #             masks.append(K.ones_like(input_i, dtype='bool'))
    #         elif K.ndim(mask_i) < K.ndim(input_i):
    #             # Mask is smaller than the input, expand it
    #             masks.append(K.expand_dims(mask_i))
    #         else:
    #             masks.append(mask_i)
    #
    #     concatenated = K.concatenate(masks, axis=self.axis)
    #     return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Intertwine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))