# lib
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.optimizers import Adam

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

    real_label: int = 1
    random_secrets: bool = False

    def initialize_model(self):

        height, width, channels = 32, 32, 3

        censorship_input = Input(shape=(height, width, channels))

        cen = Conv2D(32, (4, 4), activation='relu', padding='same')(censorship_input)
        cen = Conv2D(32, (4, 4), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(cen)

        # Block 2
        cen = Conv2D(64, (5, 5), activation='relu', padding='same')(cen)
        cen = Conv2D(64, (5, 5), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(cen)

        # Block 3
        cen = Conv2D(128, (4, 4), activation='relu', padding='same')(cen)
        cen = Conv2D(128, (4, 4), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(cen)

        # Block 4
        cen = Conv2D(256, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(256, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(cen)

        # Block 5
        cen = Conv2D(256, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(256, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(cen)

        # Classification block
        cen = Flatten(name='flatten')(cen)
        cen = Dense(1024, activation='relu', name='fc1')(cen)
        cen = Dense(1024, activation='relu', name='fc2')(cen)
        cen = Dense(1, activation='sigmoid', name='censor_prediction')(cen)

        model = Model(inputs=censorship_input, outputs=cen)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])

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
            covers, secrets = load_image_covers_and_ascii_bit_secrets(iterations_per_epoch*batch_size,
                                                                      scale=1)

            hidden_secrets = LsbSteganography.encode(covers, secrets)
            y = np.zeros(len(covers)) if self.real_label == 1 else np.ones(len(covers))

            # shuffle
            p = np.random.permutation(len(covers))
            covers, hidden_secrets, y = covers[p], hidden_secrets[p], y[p]
            covers *= 1./255.

            covers[:len(covers)//2] = hidden_secrets[:len(covers)//2]
            y[:len(covers)//2] = self.real_label

            # shuffle
            p = np.random.permutation(len(covers))
            covers, y = covers[p], y[p]

            history = self.model.fit(
                x=[covers],
                y=y,
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

