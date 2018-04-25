# system
import sys 
import os 
from dataclasses import dataclass, field

# lib
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Reshape,
    Conv1D,
    Concatenate,
    Flatten
)
from keras.activations import (
    sigmoid,
    relu,
    tanh,
)
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras.metrics import binary_accuracy
from keras.losses import mean_absolute_error

import matplotlib.pyplot as plt


# self
from models.model import NeuralCryptographyModel
from data.data import gen_symmetric_encryption_data
from general.utils import join_list_valued_dictionaries, nanify_dict_of_lists

@dataclass
class GAN(NeuralCryptographyModel):

    """
    initialize the model by passing in model
    parameters described by the following declarations. 
    These parameters must be passed in the order provided.
    
    If the parameter does not have a default value,
    provide one to the initializer.

    This is a Generative Adversarial Network (GAN) for 
    training three neural networks: Alice, Bob, and Eve.

    Alice's goal is to encrypt (with a key).
    Bob's goal is to decrypt (with a key).
    Eve's goal is to break the security of the interaction.
    currently, Eve's only training to do this revolves
    around Chosen Plaintext Attachs (CPA).
    """

    message_length: int = 16
    key_length: int = 16

    def initialize_model(self):

        def cryptography_convolution(layer, name):

            l = Conv1D(2, kernel_size=4, strides=1, activation=relu, padding='same', name=name + '_conv1')(layer)
            l = Conv1D(4, kernel_size=2, strides=2, activation=relu, padding='same', name=name + '_conv2')(l)
            l = Conv1D(4, kernel_size=1, strides=1, activation=relu, padding='same', name=name + '_conv3')(l)
            l = Conv1D(1, kernel_size=1, strides=1, activation=tanh, padding='same', name=name + '_conv4')(l)
            f = Flatten(name=name + '_flatten')(l)
            return f

        key_input = Input(shape=(self.key_length,), name='key_input')
        message_input = Input(shape=(self.message_length,), name='message_input')

        alice_input = Concatenate()([key_input, message_input])
        alice_l = Dense(self.n, activation='sigmoid', name='alice_dense')(alice_input)
        alice_l = Reshape((-1, 1), name='alice_reshape')(alice_l)
        ciphertext = cryptography_convolution(alice_l, 'alice')

        bob_input = Concatenate()([ciphertext, key_input])
        bob_l = Dense(self.n, activation='sigmoid', name='bob_dense')(bob_input)
        bob_l = Reshape((-1, 1), name='bob_reshape')(bob_l)
        bob_decrypted_ciphertext = cryptography_convolution(bob_l, 'bob')

        eve_l = Dense(self.message_length, activation='sigmoid', name='eve_dense1')(ciphertext)
        eve_l = Dense(self.n, activation='sigmoid', name='eve_dense2')(eve_l)
        eve_l = Reshape((-1, 1), name='eve_reshape')(eve_l)
        eve_decrypted_ciphertext = cryptography_convolution(eve_l, 'eve')

        eve = Model(inputs=[message_input, key_input], outputs=[eve_decrypted_ciphertext])

        for layer in eve.layers:
            if 'alice' in layer.name:
                layer.trainable = False

        eve.compile(optimizer=Adam(lr=0.0008), loss=mean_absolute_error)

        def eve_loss_for_bob(y_true, y_pred):
            return (1. - mean_absolute_error(y_true, y_pred))**2

        bob = Model(inputs=[message_input, key_input], outputs=[bob_decrypted_ciphertext, eve_decrypted_ciphertext])

        # freeze the eve layers
        for layer in bob.layers:
            if 'alice' in layer.name:
                layer.trainable = True
            if 'eve' in layer.name:
                layer.trainable = False

        bob.compile(optimizer=Adam(lr=0.0008), loss=[mean_absolute_error, eve_loss_for_bob])

        self.bob = bob
        self.eve = eve

        if self.verbose > 0:
            self.print('\n The network architecture of Bob: \n')
            bob.summary()
            self.print('\n The network architecture of Eve: \n')
            eve.summary()

        return [bob, eve]

    def __post_init__(self):
        self.n = self.message_length + self.key_length
        super(GAN, self).__post_init__()

    def __call__(self,
                 epochs=50,
                 bob_prefit_epochs=10,
                 eve_postfit_epochs=50,
                 eve_minibatch_multiplier=2,
                 iterations_per_epoch=100,
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
            bob_history = self.bob.fit(
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
            bob_history = self.bob.fit(
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

    model = GAN(verbose=1)
    history = model()
    model.visualize()

