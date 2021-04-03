import numpy as np
from copy import deepcopy
from functools import reduce
from operator import add
import os

TEAM = 41
Y = [1/2, 1, 2]
PRIZE = 50
COST = -10/Y[TEAM % 3]
ATTACK_COST = -40

GAMMA = 0.999
DELTA = 0.001

HEALTH_RANGE = 5
ARROWS_RANGE = 4
MATERIAL_RANGE = 3

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROWS_VALUES = tuple(range(ARROWS_RANGE))
MATERIAL_VALUES = tuple(range(MATERIAL_RANGE))

HEALTH_FACTOR = 25  # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1  # 0, 1, 2, 3
MATERIAL_FACTOR = 1  # 0, 1, 2

NUM_ACTIONS = 11
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 4
ACTION_SHOOT = 5  # shooting a arrow
ACTION_HIT = 6  # throwing a blade
ACTION_STAY = 7
ACTION_CRAFT = 8
ACTION_GATHER = 9
ACTION_NONE = 10


def value_iteration():
    utilities = np.zeros((position, material, arrow, health))
    actions = np.zeros((position, material, arrow, health))
    iterations = -1
    delta = 100000000
    while(delta > DELTA):
        # iteration
