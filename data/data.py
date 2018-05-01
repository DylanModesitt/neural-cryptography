import numpy as np
from data.DES import des
import os
import sys
from keras.preprocessing.image import img_to_array, load_img

def gen_symmetric_encryption_data(n, msg_len=16, key_len=16):
    """
    generate data to allow the model to train for symmetric encryption.
    :param n: how much data
    :param msg_len: the length of the message
    :param key_len: the length of the key
    :return: the data as (messages, keys)
    """
    return (np.random.randint(0, 2, size=(n, msg_len))*2-1), \
           (np.random.randint(0, 2, size=(n, key_len))*2-1)


def gen_xor_data(n, length):
    """
    generate n samples of two random bit strings of
    the same length

    :param n: the number of samples
    :param length: the length of each bit string
    :return: input_1, input_2, xor result
    """
    a = np.random.randint(0, 2, size=(n, length))
    b = np.random.randint(0, 2, size=(n, length))
    xor = a ^ b

    return (a*2-1,
            b*2-1,
            xor*2-1)


def gen_broken_otp_data(n, length, key):
    """
    generate n samples of a broken one time pad
    encryption

    :param n: the number of samples
    :param length: the length of each bit string
    :param key: The key. len(key) == length
    :return: messages, encryptions
    """
    a = np.random.randint(0, 2, size=(n, length))
    print(a[0], type(a[0]))
    xor = a ^ key

    return (a*2-1,
            xor*2-1)

def gen_reduced_des_ecb_data(amt, n, length, key, rounds):
    """
    generate amt plaintext/DES ECB ciphertext pairs n times

    :param n: the number of samples
    :param length: the length of each bit string
    :param key: The key. len(key) == length
    :param rounds: The number of DES rounds to run
    :return: messages, encryptions
    """
    p = np.vstack([np.random.randint(2,size=(64)) for j in range(amt)] for i in range(int(n/amt)))
    d = des()
    c = np.vstack([d.encrypt(key, t, rounds=rounds)] for t in p)
    return (2 * p - 1, 2 * c - 1)

def gen_des_ecb_data(n, length, key, rounds):
    """
    generate n plaintext/DES ECB ciphertext pairs

    :param n: the number of samples
    :param length: the length of each bit string
    :param key: The key. len(key) == length
    :param rounds: The number of DES rounds to run
    :return: messages, encryptions
    """
    p = np.random.randint(2,size=(n, 64))
    d = des()
    c = np.vstack([d.encrypt(key, t, rounds=rounds)] for t in p)
    return (2 * p - 1, 2 * c - 1)

def gen_secure_otp_data(n, length):
    """
    generate n samples of two random bit strings of
    the same length

    :param n: the number of samples
    :param length: the length of each bit string
    :return: input_1, input_2, xor result
    """
    a = np.random.randint(0, 2, size=(n, length))
    b = np.random.randint(0, 2, size=(n, length))
    xor = a ^ b

    return (a*2-1,
            xor*2-1)

def load_images():
    """
    load images into numpy array from /images directory

    :return: a 4d array of numpy images: (shape, height, width, channels=3)
    """
    images = []
    for file in os.listdir('./data/images/'):
        if 'jpg' in file.lower() or 'png' in file.lower():
            img = load_img(os.path.join(path, file))
            image = img_to_array(img)
            images.append(image)
    return np.array(images)
