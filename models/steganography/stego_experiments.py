import os
from typing import Sequence, Tuple

# lib
import numpy as np
from keras.preprocessing.image import array_to_img
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

# self
from models.steganography.d_2 import Steganography2D
from data.data import load_image, load_images
from general.utils import bits_from_string, string_from_bits

from models.model import NeuralCryptographyModel
from models.steganography.steganography import SteganographyData
from general.utils import join_list_valued_dictionaries, balance_real_and_fake_samples
from data.data import load_image_covers_and_random_bit_secrets
from models.steganography.d_2_evaluator import *

def video_to_frames(directory, filename, frame_size=32):
    print("Converting video " + filename + " to frames")

    cap = cv2.VideoCapture(directory + filename)
    fourcc = VideoWriter_fourcc(*'XVID')

    center_location = './data/centered_video_frames/' + filename[:-4] + '/'
    scaled_location = './data/scaled_video_frames/' + filename[:-4] + '/'

    scaled_vid = cv2.VideoWriter('./data/scaled_videos/' + filename, fourcc, float(30), (frame_size, frame_size), True)

    try:
        if not os.path.exists(center_location):
            os.makedirs(center_location)
    except OSError:
        print ('Error: Creating directory of data ' + center_location)

    try:
        if not os.path.exists(scaled_location):
            os.makedirs(scaled_location)
    except OSError:
        print ('Error: Creating directory of data ' + scaled_location)

    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == 0:
            break

        num = '{0:04}'.format(currentFrame)

        # Saves image of the current frame in png file
        centered_name = center_location + num + '.png'
        scaled_name = scaled_location + num + '.png'
        x, y, _ = frame.shape
        cv2.imwrite(centered_name, frame[int((x/2) - (frame_size/2)) : int((x/2) + (frame_size/2)), int((y/2) - (frame_size/2)): int((y/2) + (frame_size/2))])
        cv2.imwrite(scaled_name, frame[:x - (x % frame_size) : x // frame_size, :y - (y % frame_size): y // frame_size])
        scaled_vid.write(frame[:x - (x % frame_size) : x // frame_size, :y - (y % frame_size): y // frame_size])

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return center_location, scaled_location


def video_in_video(helper, secret_path, cover_path, request_num):

    output_dir = "./web/static/data/video_output/"
    fourcc = VideoWriter_fourcc(*'XVID')
    cover_vid = cv2.VideoWriter(output_dir + 'cover.mp4', fourcc, float(30), (32, 32), True)
    secret_vid = cv2.VideoWriter(output_dir + 'secret.mp4', fourcc, float(30), (32, 32), True)
    hidden_vid = cv2.VideoWriter(output_dir + 'hidden.mp4', fourcc, float(30), (32, 32), True)
    revealed_vid = cv2.VideoWriter(output_dir + 'revealed.mp4', fourcc, float(30), (32, 32), True)
    combined_vid = cv2.VideoWriter(output_dir + 'combined.mp4', fourcc, float(30), (64, 64), True)

    # Iterate through all frames
    for i in range(len(secrets)):
        # Produce the 4 frames
        cover, secret = covers[i % len(covers)], secrets[i]
        hidden_secret = np.array(array_to_img(helper.hide_given_image_in_given_cover(cover, secret)))
        revealed_secret = np.array(array_to_img(helper.decode_image_in_cover(hidden_secret)))
        cover, secret = np.array(array_to_img(cover)), np.array(array_to_img(secret))

        # Write individual frames
        cover_vid.write(cover)
        secret_vid.write(secret)
        hidden_vid.write(hidden_secret)
        revealed_vid.write(revealed_secret)

        # Combine individual frames
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
    model = Steganography2D(secret_channels=1, dir='./bin/steganography_3')
    model.load()

    helper = SteganographyImageCoverWrapper(model)

    path = './data/videos/'
    for file in os.listdir(path):
        _ , _ = video_to_frames(path, file, frame_size=32)
    #video_in_video(helper)
    #
    # cover_array = []
    # secret_array = secrets
    #
    # for i in range(len(secrets)):
    #     cover = covers[i % len(covers)]
    #     cover_array.append(cover)

    print("Hiding secret")
    hidden_secrets = helper.hide_array(secret_array, cover_array)

    print("Revealing secret")
    revealed_secrets = helper.reveal_array(hidden_secrets)

    print("Writing videos")
    for i in range(len(secret_array)):
        h = np.array(array_to_img(hidden_secrets[i]))
        r = np.array(array_to_img(revealed_secrets[i]))
        hidden_vid.write(h[:,:,::-1])
        revealed_vid.write(r[:,:,::-1])

        cover = np.array(array_to_img(cover_array[i]))
        secret = np.array(array_to_img(secret_array[i]))

        # Combine individual frames
        frameTop = np.concatenate((secret, cover), axis=1)
        frameBottom = np.concatenate((h, r), axis=1)
        frame = np.concatenate((frameTop, frameBottom), axis=0)
        frame = frame[:,:,::-1]
        combined_vid.write(frame)


def picture_in_picture(helper, secret, cover, request_num):
    cover, secret = np.array(cover), np.array(secret)

    hidden_secret = helper.hide_array([secret], [cover])
    revealed_secret = array_to_img(helper.decode_image_in_cover(hidden_secret[0]))
    hidden_secret, cover, secret = array_to_img(hidden_secret[0]), array_to_img(cover), array_to_img(secret)

    hidden_secret.save('./web/static/data/image_output/hidden' + str(request_num) + '.png', 'PNG')
    revealed_secret.save('./web/static/data/image_output/revealed' + str(request_num)  + '.png', 'PNG')


def text_in_picture(helper, string, cover, request_num):
    cover = np.array(cover)
    hidden_secret = helper.hide_str_in_image_cover(string, cover)
    revealed_secret = helper.decode_str_in_cover(hidden_secret)
    hidden_secret, cover = array_to_img(hidden_secret), array_to_img(cover)
    hidden_secret.save('./web/static/data/image_output/hidden' + str(request_num) + '.png', 'PNG')
    return revealed_secret

"""
model = Steganography2D(dir='./bin/2018-05-03_13:02__61857339')
model.load()
helper = SteganographyImageCoverWrapper(model)
secret, cover = helper.hide_str_in_random_image_cover("You a broke lad, innit?")
print(helper.decode_str_in_cover(cover)) """
