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
from models.steganography.d_2 import Steganography2D
from data.data import load_images
from general.utils import bits_from_string, string_from_bits

from models.model import NeuralCryptographyModel
from models.steganography.steganography import SteganographyData
from general.utils import join_list_valued_dictionaries, balance_real_and_fake_samples
from data.data import load_image_covers_and_random_bit_secrets


class SteganographyImageCoverWrapper:

    def __init__(self,
                 steg_2d_model: Steganography2D,
                 image_dir='./data/images/'):

        self.model = steg_2d_model
        self.images = load_images(path=image_dir,
                                  scale=steg_2d_model.image_scale)

    def hide_in_random_image_cover(self, secret):

        cover = np.random.choice(self.images)
        hidden_secret = self.model.hide(cover, secret)
        return hidden_secret

    def hide_str_in_random_image_cover(self, s):

        bits = bits_from_string(s)
        desired_len = self.model.cover_height * self.model.cover_width * self.model.secret_channels

        if len(bits) < desired_len:
            bits + [0]*(len(bits) - desired_len)

        bits = bits[:desired_len]

        bits = np.array(bits)
        bits = bits.reshape((self.model.cover_height, self.model.cover_width, self.model.secret_channels))

        return self.hide_in_random_image_cover(bits)

    def decode_str_in_cover(self, cover):

        secret = np.round(self.model.reveal(cover))
        return string_from_bits(list(secret.flatten()))


if __name__ == '__main__':

    model = Steganography2D(dir='./bin/steganography_2')
    model.load()

    helper = SteganographyImageCoverWrapper(model)

    hidden_secret = helper.hide_str_in_random_image_cover('hello')
    print(helper.decode_str_in_cover(hidden_secret))





