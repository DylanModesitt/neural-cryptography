# system
import os
from typing import Sequence, Tuple

# lib
import numpy as np
from keras.preprocessing.image import array_to_img

# self
from models.steganography.d_2 import Steganography2D
from data.data import load_image, load_images
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
                                  scale=steg_2d_model.cover_scale)

    def hide_array(self, secret_array, cover_array):
        return self.model.hide_array(np.array(cover_array), np.array(secret_array))

    def reveal_array(self, hidden_array):
        return self.model.reveal_array(hidden_array)

    def hide_in_image_cover(self, secret, cover):

        hidden_secret = self.model.hide(cover, secret)

        return hidden_secret

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

    def hide_str_in_image_cover(self, s, cover):

        bits = bits_from_string(s)
        desired_len = self.model.cover_height * self.model.cover_width * self.model.secret_channels

        if len(bits) < desired_len:
            bits += list(np.random.randint(0, 2, size=(desired_len - len(bits),)))

        bits = bits[:desired_len]
        bits = np.array(bits)
        bits = bits.reshape((self.model.cover_height, self.model.cover_width, self.model.secret_channels))

        return self.hide_in_image_cover(bits, cover)

    def hide_str_in_random_image_cover(self, s, return_cover=True):

        bits = bits_from_string(s)
        desired_len = self.model.cover_height * self.model.cover_width * self.model.secret_channels

        if len(bits) < desired_len:
            bits += list(np.random.randint(0, 2, size=(desired_len - len(bits),)))

        bits = bits[:desired_len]
        bits = np.array(bits)
        bits = bits.reshape((self.model.cover_height, self.model.cover_width, self.model.secret_channels))

        return self.hide_in_random_image_cover(bits, return_cover=return_cover)

    def decode_image_in_cover(self, cover):
        secret = self.model.reveal(cover)
        return secret

    def decode_str_in_cover(self, cover):

        secret = np.round(self.model.reveal(cover))
        return string_from_bits(list(secret.flatten().astype(bool).astype(int)))



def text_in_img_example():

    model = Steganography2D(secret_channels=1, dir='bin/text-in-image32x32')
    model.load()

    wrapper = SteganographyImageCoverWrapper(model)

    hidden_secret, cover = wrapper.hide_str_in_random_image_cover('hi how are you today ? ')
    print(hidden_secret)
    resulting_str = wrapper.decode_str_in_cover(hidden_secret)

    print('resulting str:', resulting_str)
    hidden_secret, cover = array_to_img(hidden_secret), array_to_img(cover)
    hidden_secret.show()
    cover.show()


def image_in_image_example():

    model = Steganography2D(secret_channels=3, dir='bin/image-in-image32x32')
    model.load()

    wrapper = SteganographyImageCoverWrapper(model)
    hidden_secret, cover, secret = wrapper.hide_image_in_image()
    revealed_secret = array_to_img(wrapper.decode_image_in_cover(hidden_secret))

    hidden_secret, cover, secret = array_to_img(hidden_secret), array_to_img(cover), array_to_img(secret)

    hidden_secret.show()
    cover.show()
    secret.show()
    revealed_secret.show()

if __name__ == '__main__':

    image_in_image_example()





