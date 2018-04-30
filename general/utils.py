# system
import random

# lib
import numpy as np


def generate_nonce(length=8):
    """
    Generate a psuedo ranndom number for a nonnce
    :param length: the length of the number
    :return: the nonce as a string
    """
    return ''.join([str(random.randint(0, 9)) for i in range(length)])


def join_list_valued_dictionaries(*dicts):
    """
    join two dictionaries where thhe values are lists
    by returning a result dictionary with the same keys
    and values of the concatenation of the lists a[key] + b[key].
    If the key is not shared in both dictionaries, just the (single) list
    is present

    :param a: the first dictionary
    :param b: the second dictionary
    :return: the described combination
    """

    if len(dicts) == 0:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    def sub_join(a, b):
        r = a
        for k, v in b.items():
            r[k] = r.get(k, []) + v
        return r

    return join_list_valued_dictionaries(*[sub_join(dicts[0], dicts[1]), *(dicts[2:] if len(dicts) > 2 else [])])


def nanify_dict_of_lists(dict_):
    """
    given a dictionary with list values, return the same dictionary
    with all the values of the list replaced with float('nan').

    (specific, i know)

    :param dict_: the dict
    :return: the described dictionary
    """
    return {k: [float('nan')]*len(v) for k, v in dict_.items()}


def replace_encryptions_with_random_entries(P, C, fraction=1/2, real_label=0):

    split = len(P) // (int(1/fraction))
    Y = np.ones(len(P)) if real_label == 1 else np.zeros(len(P))
    C[:split] = (np.random.randint(0, 2, size=(split, P.shape[1]))*2)-1
    Y[:split] = 0 if real_label == 1 else 1

    np.set_printoptions(edgeitems=16)
    # re-shuffle
    p = np.random.permutation(len(P))
    P = P[p]
    C = C[p]
    Y = Y[p]

    return P, C, Y

