import numpy as np


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
