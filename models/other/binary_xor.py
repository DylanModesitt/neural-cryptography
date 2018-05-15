# system
from dataclasses import dataclass

# lib
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Concatenate,
    Activation
)

from keras.optimizers import Adam

# self
from general.layers import BinaryDense
from general.binary_ops import binary_tanh

from models.model import NeuralCryptographyModel
from data.data import gen_xor_data
from general.utils import join_list_valued_dictionaries


@dataclass
class BinaryXOR(NeuralCryptographyModel):
    """
    a model that learns the xor bitwise functionn.
    """

    input_length: int = 16
    latent_dim: int = 20

    def initialize_model(self):

        input_1 = Input(shape=(self.input_length,), name='input_1')
        input_2 = Input(shape=(self.input_length,), name='input_2')

        binary_dense = BinaryDense(self.input_length)(Concatenate()([
            input_1, input_2
        ]))

        binary_dense = Activation(binary_tanh)(binary_dense)

        # binary_dense = BinaryDense(self.input_length, activation=binary_tanh)(binary_dense)

        model = Model(inputs=[input_1, input_2], outputs=binary_dense)
        model.compile(optimizer=Adam(), loss='squared_hinge')

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

            print(a[0])
            print(b[0])
            print(xor[0])

            history = self.model.fit(
                x=[a, b],
                y=xor,
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose,
            ).history

            p = [np.array([a[0]]), np.array([b[0]])]
            print(self.model.predict(p))

            xor_histories.append(history)

        history = self.generate_cohesive_history({
            'xor model': join_list_valued_dictionaries(*xor_histories),
        })

        self.save()
        self.history = history

        return history


if __name__ == '__main__':
    model = BinaryXOR(verbose=1)
    history = model(epochs=10)
    model.visualize()

