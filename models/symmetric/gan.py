# system
from enum import Enum
from abc import ABC, abstractmethod

# lib
from keras.layers import Input
from dataclasses import dataclass

# self
from models.model import NeuralCryptographyModel
from data.data import gen_symmetric_encryption_data
from general.utils import replace_encryptions_with_random_entries


class DiscriminatorGame(Enum):
    """
    An enumerated list of options for the game that the
    discriminator play's in the following GAN setup.

    ******** Optionns

    DetectEncryption: Given a plaintext, P and text C',
    is P == ENC(C'). This definition is inspired
    largely by the standard GAN setup in many deep learning
    applications.

    CiphertextIndistinguishability: Given two plaintexts P_1,
    and P_2 and two ciphertexts C_1 and C_2: which C belongs
    to which P? The exact formatting in the network is described
    below.
    """

    DetectEncryption = 0
    CiphertextIndistinguishability = 1


@dataclass
class GAN(NeuralCryptographyModel, ABC):
    """
    A ABC (Abstract Base Class) Generative Adversarial Network
    setup to perform symmetric key encryption where a
    generator (Alice and Bob) wish to communicate securely such
    that a discriminator (Eve) can not distinguish any aspect
    of the plaintext given the ciphertext.

    This model differ's from the MultiAgent model also found in
    this project because it does not ask that eve learn to actually
    encrypt but simply tell binary information given some combination
    of plaintexts annd ciphertexts.

    ******** Paramaters

    message_length: the length of the messages/ciphertexts

    key_length: the length of the symmetric key

    discrimination_mode: the kind of game that the discriminator
    is asked to play.

    """

    message_length: int = 16
    key_length: int = 16
    discrimination_mode: DiscriminatorGame = DiscriminatorGame.DetectEncryption

    def initialize_model(self):
        """
        Initialize the GAN by the given rules and configuration.

        :return: a list of the form [generator, discriminator]
        """

        message_input = Input(shape=(self.message_length,), name='message_input')
        key_input = Input(shape=(self.key_length,), name='key_input')

        if self.discrimination_mode == DiscriminatorGame.DetectEncryption:
            discriminator_message_input = Input(shape=(self.message_length,), name='discriminator_message_input')
            possible_ciphertext = Input(shape=(self.key_length,), name='possible_ciphertext_input')

            discriminator = self.initialize_discriminator(discriminator_message_input, possible_ciphertext)
        else:
            discriminator_message_input_1 = Input(shape=(self.message_length,), name='discriminator_message_input_1')
            possible_ciphertext_1 = Input(shape=(self.key_length,), name='possible_ciphertext_input_1')

            discriminator_message_input_2 = Input(shape=(self.message_length,), name='discriminator_message_input_2')
            possible_ciphertext_2 = Input(shape=(self.key_length,), name='possible_ciphertext_input_2')

            discriminator = self.initialize_discriminator(discriminator_message_input_1,
                                                          discriminator_message_input_2,
                                                          possible_ciphertext_1,
                                                          possible_ciphertext_2)

        # the discriminator should now be compiled. Thus, we can freeze
        # it and give it to the generator initialization.

        frozen_discriminator = discriminator
        for layer in frozen_discriminator:
            layer.trainable = False

        self.generator = self.initialize_discriminator(frozen_discriminator, message_input, key_input)

        return [self.generator, self.discriminator]


    @abstractmethod
    def initialize_generator(self, discriminator, *inputs):
        """
        initialize your generator given the inputs (and discriminator).
        inputs are formatted as a list of keras input layers. The number
        of layers and their meaning will be different based on the
        discrimination_mode.

        The discriminator is given to accommodate the common case
        where the generator's loss is at least somewhat determined
        by the output of the generator on the currently acting
        discriminator

        ******** DiscriminatorGame.DetectEncryption

        the discriminator will be given purely
            message_input, key_input
        and be asked to have, as an output.
            decrypted_text
        which is self explanitory. The return type also
        exectes another keras model that performs the
        encryption (for the discriminator's purpose)

        ******** DiscriminatorGame.CiphertextIndistinguishability

        same as above

        :param inputs: the inputs as described above
        :return: a list of the compiled generator and the
                 encryptor (frozen)
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_discriminator(self, *inputs):
        """
        initialize the discriminator given the inputs. inputs
        are formatted as a list of keras input layers. The number
        of layers and their meaning will be different based on the
        discrimination_mode. The discriminator must be returned
        *compiled*.

         ******** DiscriminatorGame.DetectEncryption

        the discriminator will be given purely
            plaintext_input, ciphertext_input
        and be asked to have, as an output.
            is_encryption
        which returns a 0 if the discriminator believes that that
        the given ciophertext is not an actual encryption of the given
        plaintext and 1 otherwise (if it is)

        ******** DiscriminatorGame.CiphertextIndistinguishability

        the discriminator will be given purely
            plaintext_input_1, plaintext_input_2, ciphertext_input_1,
            ciphertext_input_2
        and be asked to have, as an output.
            order
        which returns a 0 if the discriminator believes that that

            ciphertext_input_1 = ENC(plaintext_input_1)
                and
            ciphertext_input_2 = ENC(plaintext_input_2)

        and 1 if the discriminator believes that

            ciphertext_input_1 = ENC(plaintext_input_2)
                and
            ciphertext_input_2 = ENC(plaintext_input_1)

        there are no other options for the game.

        :param generator: the generator
        :param inputs: the inputs as described above
        :return: the uncompiled discriminator
        """
        raise NotImplementedError()

    @staticmethod
    def get_supported_discrimination_modes():
        """
        A method to validate what kinds of discrimination games
        the current GAN implementation supports. The default
        is that all implementations are supported.

        :return: a set of the supported DiscriminatorGame(s)
        """
        return set([v for v in DiscriminatorGame])

    def __post_init__(self):
        """
        see parent class. This GAN also validates the
        parameters of the class by checking that the given
        discrimination_mode is supported by this instance.
        :return:
        """
        self.n = self.message_length + self.key_length
        if self.discrimination_mode not in self.get_supported_discrimination_modes():
            raise ValueError('this GAN does not currently support the give Discriminator Mode')

        super(GAN, self).__post_init__()

    def fit_discriminator(self,
                          iterations,
                          batch_size):

        if self.discrimination_mode == DiscriminatorGame.DetectEncryption:

            msgs, keys = gen_symmetric_encryption_data(iterations *
                                                       batch_size,
                                                       self.message_length)

            return self.discriminator.fit(
                x=[msgs, keys],
                y=[msgs, msgs],
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose
            ).history

        else:
            raise NotImplementedError('this GAN mode is not yet supported')

    def fit_generator(self,
                      iterations,
                      batch_size):

        if self.discrimination_mode == DiscriminatorGame.DetectEncryption:

            msgs, keys = gen_symmetric_encryption_data(iterations *
                                                       batch_size,
                                                       self.message_length)

            results = self.encryptor.predict([msgs, keys])
            P, C, Y = replace_encryptions_with_random_entries(msgs, results)

            return self.discriminator.fit(
                x=[P, C],
                y=Y,
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose
            ).history

        else:
            raise NotImplementedError('this GAN mode is not yet supported')

    def __call__(self,
                 epochs=50,
                 generator_prefit_epochs=10,
                 discriminator_postfit_epochs=50,
                 discriminator_minbatch_multiplier=2,
                 iterations_per_epoch=100,
                 batch_size=512):
        """
        fit the model around randomly generated message/key pairs. The
        generator may be prefit as well as the discriminator be postfit.

        :param epochs: the number of epochs to adversarially train for
        :param generator_prefit_epochs: the number of epochs to pre-fit the generator for
        :param discriminator_postfit_epochs: the number of epochs to post-fit discriminator for
        :param discriminator_minbatch_multiplier: the minibatch multiplier for the discriminnator
        :param iterations_per_epoch: the iterations per epoch
        :param batch_size: the batch size
        :return: a dictionary of agent keys and concatenated training history values.
                 when an agent is not fit while another is, their history is populated with
                 float('nan') values.
        """

        # TODO 
        pass



