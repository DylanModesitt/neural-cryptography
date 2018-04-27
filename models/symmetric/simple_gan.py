# system
from typing import Sequence

# lib
from keras.layers import (
    Dense,
    Flatten
)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_absolute_error

# self
from general.layers import ElementWise
from models.symmetric.gan import GAN, DiscriminatorGame
from data.data import gen_symmetric_encryption_data
from general.utils import nanify_dict_of_lists

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
    alice_bitwise_latent_dims: Sequence[int] = [8, 1]
    bob_bitwise_latent_dims: Sequence[int] = [8, 1]
    tie_alice_and_bob = False

    eve_bitwise_latent_dims: Sequence[int] = [32, 8, 1]
    eve_latent_dim: Sequence[int] = [32]

    def initialize_discriminator(self, *inputs):

        if len(inputs) != 2:
            raise ValueError('this discriminator is supposed to get two inputs.'
                             'the message and the key.')

        message_input = inputs[0]
        possible_ciphertext_input = inputs[1]

        bitwise_function = Flatten()(
            ElementWise(self.eve_bitwise_latent_dim, activation='tanh')([
                message_input,
                possible_ciphertext_input
            ])
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
        return model

    def initialize_generator(self, discriminator, *inputs):

        if len(inputs) != 2:
            raise ValueError('this generator is supposed to get two inputs.'
                             'the message and the key.')

        message_input, key_input = inputs

        alice_encryption = Flatten()(
            ElementWise(self.alice_bitwise_latent_dims, activation='tanh')([
                message_input,
                key_input
            ])
        )

        bob_decryption = Flatten()(
            ElementWise(self.bob_bitwise_latent_dims, activation='tanh')([
                alice_encryption,
                key_input
            ])
        )

        for layer in discriminator:
            layer.trainable = False

        eves_opinion = discriminator(message_input, bob_decryption)

        alice = Model(inputs=inputs, outputs=alice_encryption)
        alice_bob = Model(inputs=inputs, outputs=[bob_decryption,eves_opinion])

        return alice_bob, alice

    def compile_generator(self, generator):

        def eve_prediction_loss(y_true, y_pred):
            return y_pred

        generator.compile(optimizer=Adam(lr=0.0008), loss=[mean_absolute_error, eve_prediction_loss])
        return generator

    def compile_discriminator(self, discriminator):
        discriminator.compile(optimizer=Adam(), loss='binary_crossentropy')
        return discriminator

    @staticmethod
    def get_supported_discrimination_modes():
        return [DiscriminatorGame.DetectEncryption]

