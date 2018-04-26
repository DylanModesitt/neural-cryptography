# system
from enum import Enum
from abc import ABC, abstractmethod

# lib
from keras.layers import Input

# self
from models.model import NeuralCryptographyModel


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

        generator = self.initialize_generator(message_input, key_input)

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

        generator = self.compile_generator(generator, discriminator)
        discriminator = self.compile_discriminator(discriminator, generator)

        return [generator, discriminator]


    @abstractmethod
    def initialize_generator(self, *inputs):
        """
        initialize your generator given the inputs. inputs
        are formatted as a list of keras input layers. The number
        of layers and their meaning will be different based on the
        discrimination_mode

        ******** DiscriminatorGame.DetectEncryption

        the discriminator will be given purely
            message_input, key_input
        and be asked to have, as an output.
            ciphertext, decrypted_text
        which return self-explanitory values

        ******** DiscriminatorGame.CiphertextIndistinguishability

        same as above

        :param inputs: the inputs as described above
        :return: the uncompiled generator
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_discriminator(self, *inputs):
        """
        initialize the discriminator given the inputs. inputs
        are formatted as a list of keras input layers. The number
        of layers and their meaning will be different based on the
        discrimination_mode

         ******** DiscriminatorGame.DetectEncryption

        the discriminator will be given purely
            plaintext_input, ciphertext_input
        and be asked to have, as an output.
            is_encryption
        which returns a 0 if the discriminator believes that that
        the given ciophertext is an actual encryption of the given
        plaintext

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

        :param inputs: the inputs as described above
        :return: the uncompiled discriminator
        """
        raise NotImplementedError()

    @abstractmethod
    def compile_discriminator(self, discriminator, generator):
        """
        Compile the discriminator. You may use the output of the
        *uncompiled* generator as an element of your loss. Be sure to
        freeze the generator if this is your intention

        :param discriminator: the discriminator
        :param generator: the generator
        :return: the compiled discriminator
        """
        raise NotImplementedError()

    @abstractmethod
    def compile_generator(self, discriminator, generator):
        """
        compile the generator. You may use the ouput of the
        *uncompiled* discriminator as an element of your loss.
        Be sure to freeze the generator if this is your intention.

        :param discriminator: the discriminator
        :param generator: the generator
        :return: the compiled generator
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
        raise NotImplementedError()


