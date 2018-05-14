# system
from typing import Sequence

# lib
import keras.backend as K
from dataclasses import dataclass
from keras.layers import (
    Dense,
    Conv1D,
    Concatenate,
    Reshape,
    Flatten
)

from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, binary_crossentropy

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
    alice_bitwise_latent_dims: Sequence[int] = (12, 1)
    bob_bitwise_latent_dims: Sequence[int] = (12, 1)
    alice_share_bitwise_weights: bool = True
    bob_share_bitwise_weights: bool = True

    tie_alice_and_bob = True

    eve_bitwise_latent_dims: Sequence[int] = (32, 8, 1)
    eve_latent_dim: Sequence[int] = (32,)
    eve_share_bitwise_weights: bool = False

    @staticmethod
    def cryptography_convolution(layer, name):

        l = Conv1D(2, kernel_size=4, strides=1, activation=relu, padding='same', name=name + '_conv1')(layer)
        l = Conv1D(4, kernel_size=2, strides=2, activation=relu, padding='same', name=name + '_conv2')(l)
        l = Conv1D(4, kernel_size=1, strides=1, activation=relu, padding='same', name=name + '_conv3')(l)
        l = Conv1D(1, kernel_size=1, strides=1, activation=tanh, padding='same', name=name + '_conv4')(l)
        f = Flatten(name=name + '_flatten')(l)
        return f

    def initialize_discriminator(self, *inputs):

        if len(inputs) != 2:
            raise ValueError('this discriminator is supposed to get two inputs.'
                             'the message and the key.')

        message_input = inputs[0]
        possible_ciphertext_input = inputs[1]

        alice_input = Concatenate()([key_input, message_input])
        alice_l = Dense(self.n, activation='sigmoid', name='alice_dense')(alice_input)
        alice_l = Reshape((-1, 1), name='alice_reshape')(alice_l)
        ciphertext = self.cryptography_convolution(alice_l, 'alice')

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

            enc_dec = ElementWise(self.alice_bitwise_latent_dims, activation='tanh',
                                  share_element_weights=self.alice_share_bitwise_weights)

            alice_encryption = Flatten()(
                enc_dec([
                    message_input,
                    key_input
                ])
            )

            bob_decryption = Flatten(name='decryption')(
                enc_dec([
                    alice_encryption,
                    key_input
                ]),
            )

        else:

            alice_encryption = Flatten(name='decryption')(
                ElementWise(self.bob_bitwise_latent_dims, activation='tanh',
                            share_element_weights=self.bob_share_bitwise_weights)([
                    message_input,
                    key_input
                ]),
            )

            bob_decryption = Flatten(name='decryption')(
                ElementWise(self.bob_bitwise_latent_dims, activation='tanh',
                            share_element_weights=self.bob_share_bitwise_weights)([
                    alice_encryption,
                    key_input
                ]),
            )

        eves_opinion = discriminator([message_input, alice_encryption])

        def decryption_accuracy(y_true, y_pred):
            return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

        alice = Model(inputs=inputs, outputs=alice_encryption)
        alice_bob = Model(inputs=inputs, outputs=[bob_decryption, eves_opinion])

        def discriminator_loss(y_true, y_pred):
            return K.abs(0.5 - K.mean(y_pred))

        alice_bob.compile(optimizer=Adam(),
                          loss=[mean_absolute_error, binary_crossentropy],
                          metrics=[decryption_accuracy])

        return alice_bob, alice

    @staticmethod
    def get_supported_discrimination_modes():
        return [DiscriminatorGame.DetectEncryption]


if __name__ == '__main__':

    model = EncryptionDetectionGAN(alice_share_bitwise_weights=True,
                                   bob_share_bitwise_weights=True)

    model(generator_prefit_epochs=2,
          epochs=15,
          generator_minbatch_multiplier=2,
          discriminator_postfit_epochs=10)

    model.visualize()

