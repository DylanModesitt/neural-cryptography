# system
from typing import Sequence

# lib
from dataclasses import dataclass


@dataclass
class SteganographyData:
    """
    a group of steganography data. C_*
    is the sequence of covers and S_* is
    the sequence of secrets to encode.

    These sequences may be of more generalizable
    types like arrays, images, text, etc.
    """

    C_train: Sequence
    S_train: Sequence

    C_validate: Sequence
    S_validate: Sequence
