# system
import sys
import os
from dataclasses import dataclass, field

# lib
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input,
    LocallyConnected1D,
    TimeDistributed,
    Embedding,
    LSTM,
    Dense,
    LeakyReLU,
    Reshape,
    Conv1D,
    Concatenate,
    Flatten
)
from keras.activations import (
    sigmoid,
    relu,
    tanh,
)
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras.metrics import binary_accuracy
from keras.losses import mean_absolute_error

import matplotlib.pyplot as plt

# self
from general.layers import ElementWise
from models.model import NeuralCryptographyModel
from data.data import gen_xor_data
from general.utils import join_list_valued_dictionaries, nanify_dict_of_lists


@dataclass
class XOR(NeuralCryptographyModel):
    """
    a model that learns the xor bitwise functionn.
    """

    input_length: int = 16
    latent_dim: int = 20

    def initialize_model(self):

        input_1 = Input(shape=(self.input_length,), name='input_1')
        input_2 = Input(shape=(self.input_length,), name='input_2')

        # reshape = Reshape((-1, 1))

        # bitwise_function = Flatten()(
        #     TimeDistributed(
        #         Dense(1, activation='tanh')
        #     )(LocallyConnected1D(
        #             self.latent_dim,
        #             kernel_size=2,
        #             strides=2,
        #             activation='relu'
        #         )(reshape(
        #             Intertwine()([
        #                 input_1,
        #                 input_2
        #             ]))
        #         )
        #     )
        # )

        bitwise_function = Flatten()(
            ElementWise([self.latent_dim, 1],
                        activation=['relu', 'tanh'],
                        share_element_weights=True)([input_1, input_2])
        )

        model = Model(inputs=[input_1, input_2], outputs=bitwise_function)
        model.compile(optimizer=Adam(), loss=mean_absolute_error)

        self.model = model

        if self.verbose:
            model.summary()

        return [model]

    def __call__(self,
                 epochs=10,
                 iterations_per_epoch=1000,
                 batch_size=512):
        """
        fit the model around randomly generated input_1/input_2 pairs.

        :param epochs: the number of epochs to  train for
        :param iterations_per_epoch: the iterations per epoch
        :param batch_size: the batch size
        :return: a dictionary of agent keys and concatenated training history values.
                 when an agent is not fit while another is, their history is populated with
                 float('nan') values.
        """
        xor_histories = []

        for _ in range(epochs):

            a, b, xor = gen_xor_data(iterations_per_epoch *
                                     batch_size,
                                     self.input_length)

            history = self.model.fit(
                x=[a, b],
                y=xor,
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose,
            ).history

            xor_histories.append(history)

        history = self.generate_cohesive_history({
            'xor model': join_list_valued_dictionaries(*xor_histories),
        })

        self.save()
        self.history = history

        return history


if __name__ == '__main__':
    model = XOR(verbose=1)
    history = model(epochs=10)
    model.visualize()

