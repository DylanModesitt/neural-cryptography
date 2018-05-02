# lib
import keras.backend as K
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from keras.layers import (
    Input,
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

    real_label: int = 1
    random_secrets: bool = False

    def initialize_model(self):

        height, width, channels = 32, 32, 3

        censorship_input = Input(shape=(height, width, channels))

        d_small, d_medium, d_large = (3, 4, 5)
        conv_params = {
            'padding': 'same',
            'activation': 'relu'
        }

        reveal_conv_small = Conv2D(32, kernel_size=d_small, **conv_params)(censorship_input)
        reveal_conv_small = Conv2D(32, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(32, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(32, kernel_size=d_small, **conv_params)(reveal_conv_small)

        reveal_conv_medium = Conv2D(32, kernel_size=d_medium, **conv_params)(censorship_input)
        reveal_conv_medium = Conv2D(32, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(32, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(32, kernel_size=d_medium, **conv_params)(reveal_conv_medium)

        reveal_conv_large = Conv2D(32, kernel_size=d_large, **conv_params)(censorship_input)
        reveal_conv_large = Conv2D(32, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(32, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(32, kernel_size=d_large, **conv_params)(reveal_conv_large)

        reveal_cat = Concatenate()([reveal_conv_small, reveal_conv_medium, reveal_conv_large])
        reveal_conv_small = Conv2D(32, kernel_size=d_small, **conv_params)(reveal_cat)
        reveal_conv_medium = Conv2D(32, kernel_size=d_medium, **conv_params)(reveal_cat)
        reveal_conv_large = Conv2D(32, kernel_size=d_large, **conv_params)(reveal_cat)
        reveal_final = Concatenate(name='revealed')([reveal_conv_small, reveal_conv_medium, reveal_conv_large])

        reveal_cover = Conv2D(filters=3, kernel_size=1, name='reconstructed_secret',
                              padding='same', activation='sigmoid')(reveal_final)

        # flatten = Flatten()(reveal_cover)
        # cen = Dense(1024, activation='relu')(flatten)
        # cen = Dense(1024, activation='relu')(cen)
        # cen = Dense(1, activation='sigmoid')(cen)

        def accuracy(y_true, y_pred):
            return K.mean(K.equal(y_true, K.round(y_pred)))

        model = Model(inputs=censorship_input, outputs=reveal_cover)
        model.compile(optimizer=Adam(), loss='mae', metrics=[accuracy])

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
            covers, secrets = load_image_covers_and_ascii_bit_secrets(iterations_per_epoch*batch_size,
                                                                      scale=1./255.)

            print(((covers*255)%2).astype(int))

            # hidden_secrets = LsbSteganography.encode(covers, secrets)

            # print(hidden_secrets[0][0][0])
            # print(secrets[0][0][0])
            #
            # print((hidden_secrets*255 % 2) == secrets)

            # y = np.zeros(len(covers)) if self.real_label == 0 else np.ones(len(covers))
            #
            # # shuffle
            # p = np.random.permutation(len(covers))
            # covers, hidden_secrets, y = covers[p], hidden_secrets[p], y[p]
            # # covers *= 1./255.
            #
            # covers[:len(covers)//2] = hidden_secrets[:len(covers)//2]
            # y[:len(covers)//2] = 0 if self.real_label == 1 else 1
            #
            # # shuffle
            # p = np.random.permutation(len(covers))
            # covers, y = covers[p], y[p]
            #
            # covers = covers % 2

            history = self.model.fit(
                x=[(covers*255)%2],
                y=[(covers*255)%2],
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

