# lib
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Concatenate
)
from keras.optimizers import Adam

# self
from models.adversarial.eve import Eve, AdversarialGame
from data.data import gen_des_ecb_data, gen_reduced_des_ecb_data
from general.utils import join_list_valued_dictionaries, replace_encryptions_with_random_entries


@dataclass
class DES_ECB(Eve):
    """
    This is an adversarial model to detect a broken encryption
    method that is DES ECB where all the data uses the same
    key.

    ******** Parameters

    message_length: the message length (and key length)
    """

    message_length: int = 64

    def initialize_model(self):

        message_input = Input(shape=(self.message_length,), name='message_input')
        possible_ciphertext_input = Input(shape=(self.message_length,), name='possible_ciphertext_input')

        m = Dense(1024, activation='tanh')(message_input)
        m = Dense(512, activation='tanh')(m)
        m = Dense(128, activation='tanh')(m)

        c = Dense(1024, activation='tanh')(possible_ciphertext_input)
        c = Dense(512, activation='tanh')(c)
        c = Dense(128, activation='tanh')(c)

        merge = Concatenate()([m, c])
        f = Dense(4096, activation='tanh')(merge)
        f = Dense(1024, activation='tanh')(f)
        f = Dense(1, activation='sigmoid')(f)

        model = Model(inputs=[message_input, possible_ciphertext_input], outputs=f)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        if self.verbose > 0:
            model.summary()

        self.model = model

        return [model]

    @staticmethod
    def get_supported_adversarial_modes():
        return [AdversarialGame.CiphertextIndistinguishability1]

    def __call__(self,
                 epochs=50,
                 iterations_per_epoch=250,
                 batch_size=128,
                 rounds=2):

        key = np.random.randint(0, 2, size=(self.message_length,))
        self.print("using key:", key)

        histories = []

        P, C = gen_reduced_des_ecb_data(1, iterations_per_epoch*batch_size,
                                       self.message_length,
                                       key,
                                       rounds)

        for i in range(0, epochs):

            print('\n epoch', i)
            """
            P, C = gen_des_ecb_data(iterations_per_epoch*batch_size,
                                       self.message_length,
                                       key,
                                       rounds)
            """

            # replace to play game
            P2, C2, Y = replace_encryptions_with_random_entries(P, C)

            history = self.model.fit(
                x=[P2, C2],
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

    model = DES_ECB()
    model(epochs=5)
    model.visualize()
