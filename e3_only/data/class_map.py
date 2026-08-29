"""DLRSD class map — THE single source of truth for class names and colours.

Everything that needs a class name, a colour, or a label remapping imports
from here. Nothing else in the codebase defines a class list or palette, so a
legend/prediction mismatch is impossible by construction.

Label conventions
-----------------
* DLRSD dense masks (train_1cmasks / full_test_1cmasks): pixel values 1..17.
* Chakraborty point masks (point_1cmasks): pixel values 1..17 = labelled
  class, 18 = unlabelled (background / no annotation).
* This framework's internal channels: 0..16 (remapped at manifest-write time).
  ``CLASS_NAMES[i]`` / ``PALETTE[i]`` are indexed by channel 0..16.

The colours are the official DLRSD colour-coded masks' RGB values
(``dlrsd/train_masks/*.png``), indexed by class id 1..17.
"""
import numpy as np

# channel 0..16 -> class name (same order as the remapped masks)
CLASS_NAMES = [
    "airplane",     # 0
    "bare soil",    # 1
    "buildings",    # 2
    "cars",         # 3
    "chaparral",    # 4
    "court",        # 5
    "dock",         # 6
    "field",        # 7
    "grass",        # 8
    "mobile home",  # 9
    "pavement",     # 10
    "sand",         # 11
    "sea",          # 12
    "ship",         # 13
    "tanks",        # 14
    "trees",        # 15
    "water",        # 16
]

NUM_CLASSES = len(CLASS_NAMES)      # 17
LABELLED = 17                       # highest DLRSD class id (1..17)
UNLABELLED = 18                     # point_1cmasks background value

# class id 1..17 -> RGB (official DLRSD colour-coded mask palette)
DLRSD_COLORS = {  # class id (1..17) -> RGB
    1: (166, 202, 240),    # airplane
    2: (128, 128, 0),      # bare soil
    3: (0, 0, 128),        # buildings
    4: (255, 0, 0),        # cars
    5: (0, 128, 0),        # chaparral
    6: (128, 0, 0),        # court
    7: (255, 233, 233),    # dock
    8: (160, 160, 164),    # field
    9: (0, 128, 128),      # grass
    10: (90, 87, 255),     # mobile home
    11: (255, 255, 0),     # pavement
    12: (255, 192, 0),     # sand
    13: (0, 0, 255),       # sea
    14: (255, 0, 192),     # ship
    15: (128, 0, 128),     # tanks
    16: (0, 255, 0),       # trees
    17: (0, 255, 255),     # water
}

# channel 0..16 -> RGB (remapped index = DLRSD id - 1)
PALETTE = np.asarray([DLRSD_COLORS[c + 1] for c in range(NUM_CLASSES)], dtype=np.uint8)
