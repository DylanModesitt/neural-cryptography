# system
import os
from typing import Sequence, Tuple

# lib
import numpy as np
from dataclasses import dataclass
from keras.layers import (
    Input,
    Conv2D,
    Concatenate,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

# self
from models.model import NeuralCryptographyModel
from models.steganography.steganography import SteganographyData
from general.utils import join_list_valued_dictionaries, balance_real_and_fake_samples
from data.data import load_image_covers_and_bit_secrets


@dataclass
class Steganography2D(NeuralCryptographyModel):
    """
    a neural steganography model to hide a secret in plain sight
    by embedding it in cover data. For example, hiding an image
    (secret) in another image (cover) while preserving most
    of the image qualities of the cover.

    In this model, the cover and the secret are two (or three)
    dimmensional data. The inputs should each be of the format
    (height, width, channels). For example, a 64x64 rgb image
    would be of form (64, 64, 3).

    the method of 'encryption' or 'decryption' is not known
    but is hidden as the internal encoding of the model.

    Furthermore, as a demonstration of how to practically hide
    steganographic images from censorship, this model can be
    trained additionally by an adversary which is attempting to
    detect if the modified cover is modified.

    ******** Paramaters

    :parameter cover_width: the width of the cover
    :parameter cover_height: the height of the cover
    :parameter cover_channels: the channels of the cover
    :parameter secret_channnels: the channels of thhe secret
    :parameter conv_filters: the number of filters in each convolution.
                             the default is 50.
    :parameter convolution_dimmensions: the dimmensions of the 3 convolutions
                                        that the data is passed between.
    :parameter beta: the loss modifier to emphasize how preserved the secret is.
                     in a loss function with L(c',s') = L1(c',c) + beta*L1(s',s).

    """

    cover_width: int = 32
    cover_height: int = 32

    # secret width/height must match.
    # in evaluation, zeros can pad

    cover_channels: int = 3
    secret_channels: int = 1

    conv_filters: int = 50
    convolution_dimmensions: Tuple[int] = (3, 4, 5)

    beta = 1
    gamma = 1

    def initialize_model(self):

        d_small, d_medium, d_large = self.convolution_dimmensions
        conv_params = {
            'padding': 'same',
            'activation': 'relu'
        }

        cover_input = Input(shape=(self.cover_height, self.cover_width, self.cover_channels))
        secret_input = Input(shape=(self.cover_height, self.cover_width, self.secret_channels))

        ################################
        # Prep Network
        ################################

        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(secret_input)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)

        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(secret_input)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)

        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(secret_input)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)

        prep_cat = Concatenate()([prep_conv_small, prep_conv_medium, prep_conv_large])
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_cat)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_cat)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_cat)
        prep_final = Concatenate(name='prepared')([prep_conv_large, prep_conv_medium, prep_conv_small])

        ################################
        # Hiding Network
        ################################

        hiding_input = Concatenate()([cover_input, prep_final])
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_input)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)

        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_input)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)

        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_input)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)

        hiding_cat = Concatenate()([hiding_conv_small, hiding_conv_medium, hiding_conv_large])
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_cat)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_cat)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_cat)
        hiding_final = Concatenate(name='hidden')([hiding_conv_small, hiding_conv_medium, hiding_conv_large])

        hidden_secret = Conv2D(filters=self.cover_channels, kernel_size=1, name='hidden_secret')(hiding_final)

        ################################
        # Reveal Network
        ################################

        reveal_input = hidden_secret
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_input)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)

        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_input)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)

        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_input)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)

        reveal_cat = Concatenate()([reveal_conv_small, reveal_conv_medium, reveal_conv_large])
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_cat)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_cat)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_cat)
        reveal_final = Concatenate(name='revealed')([reveal_conv_small, reveal_conv_medium, reveal_conv_large])

        reveal_cover = Conv2D(filters=self.secret_channels, kernel_size=1, name='reconstructed_secret',
                              padding='same', activation='sigmoid')(reveal_final)

        ################################
        # Censorship Network
        ################################

        censorship_input = Input(shape=(self.cover_height, self.cover_width, self.cover_channels))

        cen = Conv2D(32, (3, 3), activation='relu', padding='same')(censorship_input)
        cen = Conv2D(32, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(cen)

        # Block 2
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(cen)

        # Block 3
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(cen)

        # Block 4
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(cen)

        # Block 5
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = Conv2D(64, (3, 3), activation='relu', padding='same')(cen)
        cen = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(cen)

        # Classification block
        cen = Flatten(name='flatten')(cen)
        cen = Dense(1024, activation='relu', name='fc1')(cen)
        cen = Dense(1, activation='sigmoid', name='censor_prediction')(cen)

        self.censorship_model = Model(inputs=censorship_input, outputs=cen, name='censorship_detection')
        self.censorship_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])

        ################################
        # Deep Steganography Network
        ################################

        self.hiding_model = Model(inputs=[cover_input, secret_input], outputs=[hidden_secret])

        self.reveal_model = Model(inputs=[cover_input, secret_input], outputs=[reveal_cover])
        self.reveal_model.compile(
            optimizer=Adam(),
            loss=['binary_crossentropy'],
        )

        self.steganography_model = Model(inputs=[cover_input, secret_input], outputs=[hidden_secret, reveal_cover])
        self.steganography_model.compile(
            optimizer=Adam(),
            loss=['mae', 'binary_crossentropy'],
            loss_weights=[1, self.beta]
        )

        for layer in self.censorship_model.layers:
            layer.trainable = False

        self.adversarial_model = Model(inputs=[cover_input, secret_input],
                                       outputs=[hidden_secret, reveal_cover, self.censorship_model([hidden_secret])],
                                       name='seganography_and_adversary')

        self.adversarial_model.compile(
            optimizer=Adam(),
            loss=['mae', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[1, self.beta, self.gamma],
        )

        if self.verbose > 0:
            self.adversarial_model.summary()
            # plot_model(self.adversarial_model, to_file=os.path.join(self.dir, 'model.png'))

        return [self.adversarial_model]

    def __call__(self,
                 prefit_decryptionn_epochs=2,
                 steganography_epochs=10,
                 censorship_discriminator_epochs=5,
                 adversarial_epochs=10,
                 iterations_per_epoch=100,
                 batch_size=32):

        for i in range(0, prefit_decryptionn_epochs):

            self.print('epoch [ %s / %s]' % (i+1, prefit_decryptionn_epochs))
            self.print('>> generating data')
            covers, secrets = load_image_covers_and_bit_secrets(iterations_per_epoch*batch_size)

            print('>> fitting')
            reveal_history = self.reveal_model.fit(
                x=[covers, secrets],
                y=[secrets],
                batch_size=batch_size,
                epochs=1,
                verbose=self.verbose
            ).history

        for i in range(0, steganography_epochs):

            self.print('epoch [ %s / %s]' % (i+1, steganography_epochs))
            self.print('>> generating data')
            covers, secrets = load_image_covers_and_bit_secrets(iterations_per_epoch*batch_size)

            self.print('>> fitting')
            steganography_history = self.steganography_model.fit(
                x=[covers, secrets],
                y=[covers, secrets],
                batch_size=batch_size,
                epochs=1,
                verbose=self.verbose
            ).history

        for i in range(0, adversarial_epochs):

            self.print('epoch [ %s / %s]' % (i+1, steganography_epochs))

            self.print('>> generating data')

            # gen covers/secrets
            covers, secrets = load_image_covers_and_bit_secrets(iterations_per_epoch*batch_size)
            # gen steg(cover, segret)
            hidden_secrets = self.hiding_model.predict([covers, secrets])
            samples, labels = balance_real_and_fake_samples(covers, hidden_secrets)

            censorship_history = self.censorship_model.fit(
                x=[samples],
                y=[labels],
                batch_size=batch_size,
                epochs=1,
                verbose=self.verbose
            )

            self.print('>> generating data')
            covers, secrets = load_image_covers_and_bit_secrets(iterations_per_epoch*batch_size)

            self.print('>> fitting')
            adverarial_history = self.adversarial_model.fit(
                x=[covers, secrets],
                y=[covers, secrets, np.zeros(len(covers))],
                batch_size=batch_size,
                epochs=1,
                verbose=self.verbose
            ).history



        return None


if __name__ == '__main__':

    model = Steganography2D()
    model(steganography_epochs=0,
          prefit_decryptionn_epochs=0)

