# system
from dataclasses import dataclass
from typing import Sequence

# lib
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Reshape,
    Conv1D,
    Concatenate,
    Flatten
)

from keras.optimizers import Adam
from keras.losses import mean_absolute_error

# self
from models.model import NeuralCryptographyModel
from general.layers import ElementWise
from data.data import gen_symmetric_encryption_data
from general.utils import join_list_valued_dictionaries, nanify_dict_of_lists

@dataclass
class MultiAgentDecryption(NeuralCryptographyModel):

    """
    initialize the model by passing in model
    parameters described by the following declarations. 
    These parameters must be passed in the order provided.
    
    If the parameter does not have a default value,
    provide one to the initializer.

    This is a (somewhat) Generative Adversarial Network (GAN) for
    training three neural networks: Alice, Bob, and Eve.

    The model architecture is based of a paper published by
    Google Brain: https://arxiv.org/pdf/1610.06918v1.pdf

    Alice's goal is to encrypt (with a key).
    Bob's goal is to decrypt (with a key).
    Eve's goal is to break the security of the interaction.
    """

    message_length: int = 16
    key_length: int = 16

    alice_bitwise_latent_dims: Sequence[int] = (12, 1)
    bob_bitwise_latent_dims: Sequence[int] = (12, 1)
    alice_share_bitwise_weights: bool = True
    bob_share_bitwise_weights: bool = True

    tie_alice_and_bob = True

    eve_bitwise_latent_dims: Sequence[int] = (32, 8, 1)
    eve_share_bitwise_weights: bool = False

    def initialize_model(self):

        key_input = Input(shape=(self.key_length,), name='key_input')
        message_input = Input(shape=(self.message_length,), name='message_input')

        alice_encryption = Flatten(name='alice_encryption')(
            ElementWise(self.alice_bitwise_latent_dims, activation='tanh',
                        share_element_weights=self.alice_share_bitwise_weights,
                        name='alice')([
                message_input,
                key_input
            ]),
        )

        bob_decryption = Flatten(name='bob_decryption')(
            ElementWise(self.bob_bitwise_latent_dims, activation='tanh',
                        share_element_weights=self.bob_share_bitwise_weights,
                        name='bob')([
                alice_encryption,
                key_input
            ]),
        )

        eve_decryption = Flatten(name='eve_decryption')(
            ElementWise(self.eve_bitwise_latent_dims, activation='tanh',
                        share_element_weights=self.eve_share_bitwise_weights,
                        name='eve')([
                alice_encryption,
                message_input
            ]),
        )

        eve = Model(inputs=[message_input, key_input], outputs=[eve_decryption])

        for layer in eve.layers:
            if 'alice' in layer.name:
                layer.trainable = False

        eve.compile(optimizer=Adam(lr=0.0008), loss=mean_absolute_error)

        def eve_loss_for_bob(y_true, y_pred):
            return (1. - mean_absolute_error(y_true, y_pred))**2

        all = Model(inputs=[message_input, key_input], outputs=[bob_decryption, eve_decryption])

        # freeze the eve layers
        for layer in all.layers:
            if 'alice' in layer.name:
                layer.trainable = True
            if 'eve' in layer.name:
                layer.trainable = False

        all.compile(optimizer=Adam(lr=0.0008), loss=[mean_absolute_error, eve_loss_for_bob])

        self.all = all
        self.eve = eve

        if self.verbose > 0:
            self.print('\n The network architecture of Bob: \n')
            all.summary()
            self.print('\n The network architecture of Eve: \n')
            eve.summary()

        return [all, eve]

    def __post_init__(self):
        self.n = self.message_length + self.key_length
        super(MultiAgentDecryption, self).__post_init__()

    def __call__(self,
                 epochs=50,
                 bob_prefit_epochs=10,
                 eve_postfit_epochs=50,
                 eve_minibatch_multiplier=2,
                 iterations_per_epoch=2000,
                 batch_size=512):
        """
        fit the model around randomly generated message/key pairs.

        :param epochs: the number of epochs to adversarially train for
        :param bob_prefit_epochs: the number of epochs to pre-fit bob for
        :param eve_postfit_epochs: the number of epochs to post-fit bob for
        :param eve_minibatch_multiplier: the minibatch multiplier for eve
        :param iterations_per_epoch: the iterations per epoch
        :param batch_size: the batch size
        :return: a dictionary of agent keys and concatenated training history values.
                 when an agent is not fit while another is, their history is populated with
                 float('nan') values.
        """

        bob_histories = []
        eve_histories = []

        for i in range(0, bob_prefit_epochs):

            msgs, keys = gen_symmetric_encryption_data(iterations_per_epoch *
                                                       batch_size)

            self.print('pre-fitting bob')
            bob_history = self.all.fit(
                x=[msgs, keys],
                y=[msgs, msgs],
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose
            ).history

            bob_histories.append(bob_history)
            eve_histories.append(nanify_dict_of_lists(bob_history))

            self.save()

        for i in range(0, epochs):

            msgs, keys = gen_symmetric_encryption_data(iterations_per_epoch *
                                                       batch_size)

            self.print('fitting bob')
            bob_history = self.all.fit(
                x=[msgs, keys],
                y=[msgs, msgs],
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose
            ).history

            bob_histories.append(bob_history)

            msgs, keys = gen_symmetric_encryption_data(eve_minibatch_multiplier *
                                                       iterations_per_epoch *
                                                       batch_size)
            self.print('fitting eve')
            eve_history = self.eve.fit(
                x=[msgs, keys],
                y=[msgs],
                epochs=1,
                batch_size=2*batch_size,
                verbose=self.verbose
            ).history

            eve_histories.append(eve_history)

            self.save()

        for i in range(0, eve_postfit_epochs):

            msgs, keys = gen_symmetric_encryption_data(iterations_per_epoch *
                                                       eve_minibatch_multiplier *
                                                       batch_size)
            print('fitting eve')

            eve_history = self.eve.fit(
                x=[msgs, keys],
                y=[msgs],
                epochs=1,
                batch_size=2 * batch_size,
                verbose=self.verbose
            ).history

            eve_histories.append(eve_history)
            bob_histories.append(nanify_dict_of_lists(eve_history))

            self.save()

        history = self.generate_cohesive_history({
            'alice / bob': join_list_valued_dictionaries(*bob_histories),
            'eve': join_list_valued_dictionaries(*eve_histories)
        })

        self.history = history

        return history


if __name__ == '__main__':

    model = MultiAgentDecryption(verbose=1)
    history = model(bob_prefit_epochs=0)
    model.visualize()

