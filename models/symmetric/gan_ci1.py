# system
from typing import Sequence

# lib
import numpy as np
import keras.backend as K
from dataclasses import dataclass
from keras.layers import (
    Dense,
    Flatten,
    GaussianNoise,
    Conv1D,
    Concatenate,
    Reshape,
    LocallyConnected1D,
    PReLU,
    Add,
    Lambda,
)

from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, binary_crossentropy

# self
from general.layers import ElementWise
from general.binary_ops import binary_tanh
from models.symmetric.gan import GAN, DiscriminatorGame


@dataclass
class EncryptionDetectionGAN(GAN):
    """
    A simple Generative Adversarial Network (GAN)
    that implements an approach for an adversary, Eve,
    to learn about an encryption/decription technique by
    two other Neural Networks: Alice/Eve. This game is a
    sort of Ciphertext Indistinguishability game where
    Eve is expected, given a potential encryption, to detect
    whether or not that encryption actually came from the plaintext.

    That is, if the ciphertext appears entirely random given the plaintext,
    Alice and Bob will win. Otherwise, Eve should win.

    This game is somewhat based on the CPA game.
    """
    alice_bitwise_latent_dims: Sequence[int] = (20, 1)
    bob_bitwise_latent_dims: Sequence[int] = (20, 1)
    alice_share_bitwise_weights: bool = True
    bob_share_bitwise_weights: bool = True

    tie_alice_and_bob = False

    use_bias: bool = True

    eve_bitwise_latent_dims: Sequence[int] = (32, 8, 1)
    eve_latent_dim: Sequence[int] = (32,)
    eve_share_bitwise_weights: bool = False

    @staticmethod
    def cryptography_convolution(layer):

        l = Conv1D(2, kernel_size=4, strides=1, activation='relu', padding='same')(layer)
        l = Conv1D(4, kernel_size=2, strides=2, activation='relu', padding='same')(l)
        l = Conv1D(4, kernel_size=1, strides=1, activation='relu', padding='same')(l)
        l = Conv1D(1, kernel_size=1, strides=1, activation='relu', padding='same')(l)
        f = Flatten()(l)
        return f

    def initialize_discriminator(self, *inputs):

        if len(inputs) != 2:
            raise ValueError('this discriminator is supposed to get two inputs.'
                             'the message and the key.')

        message_input = inputs[0]
        possible_ciphertext_input = inputs[1]

        bitwise_function = Flatten()(
            ElementWise(
                self.eve_bitwise_latent_dims,
                activation='tanh',
                share_element_weights=self.eve_share_bitwise_weights)([message_input,
                                                                       possible_ciphertext_input])
        )

        dense = Dense(
            self.eve_latent_dim[0],
            activation='relu'
        )(bitwise_function)

        for units in self.eve_latent_dim[1:]:
            dense = Dense(
                units,
                activation='relu'
            )(dense)

        pred = Dense(
            1,
            activation='sigmoid'
        )(dense)

        model = Model(inputs=inputs, outputs=pred)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])

        return model

    def initialize_generator(self, discriminator, *inputs):

        if len(inputs) != 2:
            raise ValueError('this generator is supposed to get two inputs.'
                             ' the message and the key.')

        message_input, key_input = inputs

        if self.tie_alice_and_bob:

            if self.alice_share_bitwise_weights != self.bob_share_bitwise_weights:
                raise ValueError('bob and alice are tied however they do not '
                                 'have the same weight sharing settings.')

            enc_dec = ElementWise(self.alice_bitwise_latent_dims, activation=['relu', binary_tanh],
                                  share_element_weights=self.alice_share_bitwise_weights,
                                  use_bias=self.use_bias)

            alice_encryption = Flatten(name='encryption')(
                enc_dec([
                    key_input,
                    message_input
                ])
            )

            modified_encryption = GaussianNoise(0.1)(alice_encryption)

            bob_decryption = Flatten(name='decryption')(
                enc_dec([
                    key_input,
                    modified_encryption,
                ]),
            )

        else:

            alice_input = Concatenate()([key_input, message_input])
            alice_l = Dense(self.n, activation='sigmoid', name='alice_dense')(alice_input)
            alice_l = Reshape((-1, 1), name='alice_reshape')(alice_l)
            alice_encryption = self.cryptography_convolution(alice_l)

            bob_input = Concatenate()([alice_encryption, key_input])
            bob_l = Dense(self.n, activation='sigmoid', name='bob_dense')(bob_input)
            bob_l = Reshape((-1, 1), name='bob_reshape')(bob_l)
            bob_decryption = self.cryptography_convolution(bob_l)


        eves_opinion = discriminator([message_input, alice_encryption])

        alice = Model(inputs=inputs, outputs=alice_encryption)
        self.alice = alice
        alice_bob = Model(inputs=inputs, outputs=[alice_encryption, bob_decryption, eves_opinion])
        self.alice_bob = alice_bob

        # 1 - K.mean(K.abs(y_pred), axis=-1) +
        def bit_encforcing_loss(y_true, y_pred):
            return K.abs(K.mean(y_pred))

        def decryption_accuracy(y_true, y_pred):
            return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

        def indistinguishable_loss(y_true, y_pred):
            return K.abs(0.5 - K.mean(y_pred))

        alice_bob.compile(optimizer=Adam(lr=0.0001),
                          loss=[bit_encforcing_loss, mean_absolute_error, indistinguishable_loss],
                          loss_weights=list(self.generator_loss_weights),
                          metrics=[decryption_accuracy])

        return alice_bob, alice

    @staticmethod
    def get_supported_discrimination_modes():
        return [DiscriminatorGame.DetectEncryption]


def trial():

    model = EncryptionDetectionGAN(dir='./bin/gan_ci1_sucess1')
    model.load()

    # -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

    print(model.alice.predict([np.array([[
        1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]]),np.array([[
        1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]])]))


if __name__ == '__main__':

    # trial()

    model = EncryptionDetectionGAN(alice_share_bitwise_weights=True,
                                   bob_share_bitwise_weights=True)

    model(generator_prefit_epochs=0,
          epochs=10,
          discriminator_postfit_epochs=5)

    model.visualize()

