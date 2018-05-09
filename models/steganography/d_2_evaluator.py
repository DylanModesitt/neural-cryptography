# system
import os
from typing import Sequence, Tuple

# lib
import numpy as np
from keras.preprocessing.image import array_to_img
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

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

    def hide_given_image_in_given_cover(self, secret, cover):
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

def vid_to_frames(location):
    cap = cv2.VideoCapture(location)
    fourcc = VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./data/vid/small_vid.mp4', fourcc, float(20), (32, 32), True)
    out2 = cv2.VideoWriter('./data/vid/center_small_vid.mp4', fourcc, float(20), (32, 32), True)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(True):
        if not(currentFrame % 15 == 0):
            pass
           # continue

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == 0:
            break

        num = '{0:04}'.format(currentFrame)

        # Saves image of the current frame in jpg file
        name = './data/vid/png/frame' + num + '.png'
        name2 = './data/vid/pngs/frame' + num + '.png'
        #print(frame.shape)
        #print ('Creating...' + name)
        cv2.imwrite(name2, frame[344:376,624:656])
        out2.write(frame[344:376, 624:656])
        #print(frame[:704:22, ::40].shape)
        cv2.imwrite(name, frame[:704:22, ::40])
        out.write(frame[:704:22, ::40])

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def video_in_video(helper):
    #vid_to_frames('data/vid/vid.mp4')

    image_dir = 'data/vid/png'
    covers = load_images(path=image_dir + 's', scale=model.image_scale)#,
    secrets = load_images(path=image_dir, scale=model.image_scale)
    fourcc = VideoWriter_fourcc(*'XVID')
    #hidden_vid = cv2.VideoWriter('./data/vid/small_vid_covered.mp4', fourcc, float(30), (32, 32), True)
    #revealed_vid = cv2.VideoWriter('./data/vid/small_vid_revealed.mp4', fourcc, float(30), (32, 32), True)
    combined_vid = cv2.VideoWriter('./data/vid/combined_vid.mp4', fourcc, float(30), (64, 64), True)

    for i in range(len(secrets)):
        cover, secret = covers[i], secrets[i]
        hidden_secret = np.array(array_to_img(helper.hide_given_image_in_given_cover(cover, secret)))
        revealed_secret = np.array(array_to_img(helper.decode_image_in_cover(hidden_secret)))
        cover, secret = np.array(array_to_img(cover)), np.array(array_to_img(secret))
        #hidden_vid.write(hidden_secret)
        #revealed_vid.write(revealed_secret)
        frameTop = np.concatenate((np.array(array_to_img(secret)), np.array(array_to_img(cover))), axis=1)
        frameBottom = np.concatenate((hidden_secret, revealed_secret), axis=1)
        frame = np.concatenate((frameTop, frameBottom), axis=0)
        frame = frame[:,:,::-1]
        combined_vid.write(frame)

def picture_in_picture(helper):
    hidden_secret, cover, secret = helper.hide_image_in_image(return_cover=True)
    revealed_secret = array_to_img(helper.decode_image_in_cover(hidden_secret))
    hidden_secret, cover, secret = array_to_img(hidden_secret), array_to_img(cover), array_to_img(secret)

    cover.save('./cover.png', 'PNG')
    secret.save('./secret.png', 'PNG')
    hidden_secret.save('./hidden_secret.png', 'PNG')
    revealed_secret.save('./revealed_secret.png', 'PNG')


if __name__ == '__main__':
    model = Steganography2D(dir='./bin/2018-05-03_13:02__61857339')
    model.load()

    helper = SteganographyImageCoverWrapper(model)

    video_in_video(helper)




