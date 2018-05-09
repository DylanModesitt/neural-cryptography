# system
import os
from typing import Sequence, Tuple

# lib
import numpy as np
from keras.preprocessing.image import array_to_img

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

    def hide_in_random_image_cover(self, secret, return_cover=True):

        idx = np.random.randint(0, len(self.images))
        cover = self.images[idx]
        hidden_secret = self.model.hide(cover, secret)

        if return_cover:
            return hidden_secret, cover

        return hidden_secret

    def hide_image_in_image(self, return_cover=True, return_secret=True):

        idx = np.random.randint(0, len(self.images))
        idx2 = np.random.randint(0, len(self.images))

        cover = self.images[idx]
        secret = self.images[idx2]

        hidden_secret = self.model.hide(cover, secret)

        if return_cover and return_secret:
            return hidden_secret, cover, secret

        return hidden_secret

    def hide_str_in_random_image_cover(self, s):

        bits = bits_from_string(s)
        desired_len = self.model.cover_height * self.model.cover_width * self.model.secret_channels

        if len(bits) < desired_len:
            bits += list(np.random.randint(0, 2, size=(desired_len - len(bits),)))

        bits = bits[:desired_len]
        bits = np.array(bits)
        bits = bits.reshape((self.model.cover_height, self.model.cover_width, self.model.secret_channels))

        return self.hide_in_random_image_cover(bits)

    def decode_image_in_cover(self, cover):
        secret = self.model.reveal(cover)
        return secret

    def decode_str_in_cover(self, cover):

        secret = np.round(self.model.reveal(cover))
        print(list(secret.flatten().astype(int)))
        return string_from_bits(list(secret.flatten().astype(int)))


if __name__ == '__main__':

    model = Steganography2D(dir='./bin/2018-05-03_13:02__61857339')
    model.load()

    helper = SteganographyImageCoverWrapper(model)

    hidden_secret, cover, secret = helper.hide_image_in_image(return_cover=True)
    revealed_secret = array_to_img(helper.decode_image_in_cover(hidden_secret))
    hidden_secret, cover, secret = array_to_img(hidden_secret), array_to_img(cover), array_to_img(secret)

    cover.save('./cover.png', 'PNG')
    secret.save('./secret.png', 'PNG')
    hidden_secret.save('./hidden_secret.png', 'PNG')
    revealed_secret.save('./revealed_secret.png', 'PNG')






