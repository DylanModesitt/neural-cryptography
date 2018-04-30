import numpy as np
#from Crypto.Cipher import DES
from data.DES import des

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

def bitstring_to_bytes(bitstring):
    return int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, 'big')

def gen_des_ecb_data(n, length, key, rounds):
    """
    TODO
    # get key in bytes
    d = ""
    for i in range(length):
        d += str(key[i])
    byte_key = bitstring_to_bytes(d)

    # create DES instance
    cipher = DES.new(byte_key, DES.MODE_ECB)

    # create plaintexts
    array_p = np.random.randint(0, 2, size=(n, length))
    P = np.zeros((n), dtype=np.bool_)
    for i in range(n):
        p = ""
        for j in range(length):
            p += str(array_p[i][j])
        P[i] = bitstring_to_bytes(p)

    C = np.vstack([cipher.encrypt(p)] for p in P)


    return (P*2-1,
            C*2-1)
"""
    p = np.random.randint(2,size=(n, 64))
    d = des()
    c = np.vstack([d.encrypt(key, t, rounds=16)] for t in p)
    return (p*2-1, c*2-1) 
