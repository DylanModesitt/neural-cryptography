# lib
import keras.backend as K
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    TimeDistributed,
    LSTM,
    Conv2D,
    Concatenate,
    AveragePooling2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense
)
from keras.optimizers import Adam
from keras.metrics import binary_accuracy

# self
from models.adversarial.eve import Eve, AdversarialGame
from general.layers import ElementWise
from data.data import (
    load_image_covers_and_ascii_bit_secrets,
    load_image_covers_and_random_bit_secrets,
    LsbSteganography
)

from general.utils import join_list_valued_dictionaries, replace_encryptions_with_random_entries


@dataclass
class LsbDetection(Eve):
    """
    This is an adversarial model to detect if an image
    contains LSB steganography where the secret is actual
    english words.

    ********* Parameters
    :parameter real_label: the label for an image that does not
                           contain steganographic content
    """

    embedding_dimmension: int = 100

    real_label: int = 1
    random_secrets: bool = False

    def initialize_model(self):

        height, width, channels = 32, 32, 3

        censorship_input = Input(shape=(height, width, channels))

        flattend_input = Flatten()(censorship_input)
        censor = Embedding(input_dim=256, output_dim=self.embedding_dimmension)(flattend_input)

        censor = TimeDistributed(
            Dense(128, activation='tanh')
        )(censor)

        censor = TimeDistributed(
            Dense(1, activation='tanh')
        )(censor)

        censor = Dense(4096, activation='tanh')(Flatten()(censor))
        censor = Dense(1024, activation='tanh')(censor)
        censor = Dense(1, activation='sigmoid')(censor)

        model = Model(inputs=censorship_input, outputs=censor)
        model.compile(optimizer=Adam(), loss='mae', metrics=['acc'])

        if self.verbose > 0:
            model.summary()

        self.model = model

        return [model]

    @staticmethod
    def get_supported_adversarial_modes():
        return [AdversarialGame.CiphertextIndistinguishability1]

    def __call__(self,
                 epochs=50,
                 iterations_per_epoch=100,
                 batch_size=32):

        histories = []
        for i in range(0, epochs):

            print('\n epoch', i)
            covers, secrets = load_image_covers_and_random_bit_secrets(iterations_per_epoch*batch_size,
                                                                      scale=1)

            hidden_secrets = LsbSteganography.encode(covers, secrets, scale=1)
            y = np.zeros(len(covers)) if self.real_label == 0 else np.ones(len(covers))

            # shuffle
            p = np.random.permutation(len(covers))
            covers, hidden_secrets, y = covers[p], hidden_secrets[p], y[p]
            # covers[:len(covers)//2] = hidden_secrets[:len(covers)//2]
            y[:len(covers)//2] = 0 if self.real_label == 1 else 1

            # shuffle
            p = np.random.permutation(len(covers))
            covers, y = covers[p], y[p]

            history = self.model.fit(
                x=[covers],
                y=[y],
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

    model = LsbDetection()
    model(epochs=10)
    model.visualize()

