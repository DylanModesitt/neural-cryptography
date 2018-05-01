# system
from typing import Sequence, Tuple

# lib
from dataclasses import dataclass
from keras.layers import (
    Input,
    Conv2D,
    Concatenate,
    BatchNormalization,
)
from keras.models import Model

# self
from models.model import NeuralCryptographyModel




@dataclass
class Steganography2D(NeuralCryptographyModel):

    cover_width: int = 64
    cover_height: int = 64

    # secret width/height must match.
    # in evaluation, zeros can pad

    cover_channels: int = 3
    secret_channels: int = 1

    conv_filters: int = 50
    convolution_dimmensions: Tuple[int] = (3, 4, 5)

    def initialize_model(self):

        d_small, d_medium, d_large = self.convolution_dimmensions
        conv_params = {
            'padding': 'same',
            'activation': 'relu'
        }

        cover_input = Input(shape=(self.cover_height, self.cover_width, self.cover_channels))
        secret_input = Input(shape=(self.cover_height, self.cover_width, self.secret_channels))

        ###################
        # Prep Network
        ##################

        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(secret_input)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_conv_small)

        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(secret_input)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_conv_medium)

        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(secret_input)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_conv_large)

        prep_cat = Concatenate()([prep_conv_small, prep_conv_medium, prep_conv_large])
        prep_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(prep_cat)
        prep_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(prep_cat)
        prep_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(prep_cat)
        prep_final = Concatenate()([prep_conv_small, prep_conv_medium, prep_conv_large])

        prep_model = Model(inputs=secret_input, outputs=prep_final)

        ###################
        # Hiding Network
        ##################

        hiding_input = Concatenate()([cover_input, prep_final])
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_input)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_conv_small)

        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_input)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_conv_medium)

        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_input)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_conv_large)

        hiding_cat = Concatenate()([hiding_conv_small, hiding_conv_medium, hiding_conv_large])
        hiding_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(hiding_cat)
        hiding_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(hiding_cat)
        hiding_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(hiding_cat)
        hiding_final = Concatenate()([hiding_conv_small, hiding_conv_medium, hiding_conv_large])

        hiding_model = Model(inputs=hiding_input, outputs=hiding_final)

        ###################
        # Reveal Network
        ##################

        reveal_input = hiding_final
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_input)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_conv_small)

        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_input)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_conv_medium)

        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_input)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_conv_large)

        reveal_cat = Concatenate()([reveal_conv_small, reveal_conv_medium, reveal_conv_large])
        reveal_conv_small = Conv2D(self.conv_filters, kernel_size=d_small, **conv_params)(reveal_cat)
        reveal_conv_medium = Conv2D(self.conv_filters, kernel_size=d_medium, **conv_params)(reveal_cat)
        reveal_conv_large = Conv2D(self.conv_filters, kernel_size=d_large, **conv_params)(reveal_cat)
        reveal_final = Concatenate()([reveal_conv_small, reveal_conv_medium, reveal_conv_large])

        reveal_model = Model(inputs=reveal_input, outputs=reveal_final)

    def __call__(self,
                 data,
                 epochs,
                 batch_size):

        pass