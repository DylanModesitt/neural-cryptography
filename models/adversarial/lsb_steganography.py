# lib
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from keras.layers import (
    Input,
    Flatten,
    Dense
)

# self
from models.adversarial.eve import Eve, AdversarialGame
from general.layers import ElementWise
from data.data import gen_broken_otp_data
from general.utils import join_list_valued_dictionaries, replace_encryptions_with_random_entries


@dataclass
class OTPSameKey(Eve):
    """
    This is an adversarial model to detect if an image
    contains LSB steganography where the secret is actual
    english words.

    """

    message_length: int = 16

    def initialize_model(self):

        message_input = Input(shape=(self.message_length,), name='message_input')
        possible_ciphertext_input = Input(shape=(self.message_length,), name='possible_ciphertext_input')

        bitwise_function = Flatten()(
            ElementWise([8, 1], activation='tanh')([
                message_input,
                possible_ciphertext_input
            ])
        )

        dense = Dense(
            self.message_length,
            activation='relu'
        )(bitwise_function)

        pred = Dense(
            1,
            activation='sigmoid'
        )(dense)

        model = Model(inputs=[message_input, possible_ciphertext_input], outputs=pred)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        self.model = model

        return [model]

    @staticmethod
    def get_supported_adversarial_modes():
        return [AdversarialGame.CiphertextIndistinguishability1]

    def __call__(self,
                 epochs=50,
                 iterations_per_epoch=300,
                 batch_size=512):

        key = np.random.randint(0, 2, size=(self.message_length,))
        self.print("using key:", key)

        histories = []
        for i in range(0, epochs):

            print('\n epoch', i)
            P, C = gen_broken_otp_data(iterations_per_epoch*
                                       batch_size,
                                       self.message_length,
                                       key)

            # replace to play game
            P, C, Y = replace_encryptions_with_random_entries(P, C)

            history = self.model.fit(
                x=[P, C],
                y=Y,
                verbose=self.verbose,
                epochs=1,
                batch_size=batch_size
            ).history

            histories.append(history)
            self.save()

        history = self.generate_cohesive_history({
            'eve': join_list_valued_dictionaries(*histories)
        })

        self.history = history

        return history


if __name__ == '__main__':

    model = OTPSameKey()
    model(epochs=10)
    model.visualize()

