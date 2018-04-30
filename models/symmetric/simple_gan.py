# system
from typing import Sequence

# lib
import keras.backend as K
from dataclasses import dataclass
from keras.layers import (
    Dense,
    Flatten
)

from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras.metrics import binary_accuracy

# self
from general.layers import ElementWise
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
    alice_bitwise_latent_dims: Sequence[int] = (8, 1)
    bob_bitwise_latent_dims: Sequence[int] = (8, 1)
    alice_share_bitwise_weights: bool = False
    bob_share_bitwise_weights: bool = False

    tie_alice_and_bob = False

    eve_bitwise_latent_dims: Sequence[int] = (32, 8, 1)
    eve_latent_dim: Sequence[int] = (32,)
    eve_share_bitwise_weights: bool = False

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

        alice_encryption = Flatten()(
            ElementWise(self.alice_bitwise_latent_dims, activation='tanh',
                        share_element_weights=self.alice_share_bitwise_weights)([
                message_input,
                key_input
            ])
        )

        bob_decryption = Flatten(name='decryption')(
            ElementWise(self.bob_bitwise_latent_dims, activation='tanh',
                        share_element_weights=self.bob_share_bitwise_weights)([
                alice_encryption,
                key_input
            ]),
        )

        eves_opinion = discriminator([message_input, alice_encryption])

        alice = Model(inputs=inputs, outputs=alice_encryption)
        alice_bob = Model(inputs=inputs, outputs=[bob_decryption, eves_opinion])

        def discriminator_loss(y_true, y_pred):
            return K.abs(0.5 - binary_accuracy(y_true, y_pred)) ** 2

        alice_bob.compile(optimizer=Adam(lr=0.0008),
                          loss=[mean_absolute_error, discriminator_loss])

        return alice_bob, alice

    @staticmethod
    def get_supported_discrimination_modes():
        return [DiscriminatorGame.DetectEncryption]


if __name__ == '__main__':

    model = EncryptionDetectionGAN(alice_share_bitwise_weights=True,
                                   bob_share_bitwise_weights=True)

    model(generator_prefit_epochs=5,
          epochs=15,
          discriminator_postfit_epochs=10)

    model.visualize()

