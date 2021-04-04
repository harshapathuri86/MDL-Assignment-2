import numpy as np
from copy import deepcopy
from functools import reduce
from operator import add
import os

TEAM = 41
Y = [1 / 2, 1, 2]
PRIZE = 50
COST = -10 / Y[TEAM % 3]
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

NUM_ACTIONS = 10

ACTIONS = {
    "ACTION_UP": 0,
    "ACTION_DOWN": 1,
    "ACTION_LEFT": 2,
    "ACTION_RIGHT": 3,
    "ACTION_SHOOT": 4,  # shooting a arrow
    "ACTION_HIT": 5,  # throwing a blade
    "ACTION_STAY": 6,
    "ACTION_CRAFT": 7,
    "ACTION_GATHER": 8,
    "ACTION_NONE": 9,
}

POSITIONS = {
    "N": 0,
    "S": 1,
    "E": 2,
    "W": 3,
    "C": 4,
}

ENEMY_STATE = {
    "D": 0,
    "R": 1,
}


POSITIONS = 5  # n, s, e, w, c
ENEMY_STATE = 0  # 0 D, 1 R


class State:
    def __init__(self, enemy_health, num_arrows, stamina):
        if (
            (enemy_health not in HEALTH_VALUES)
            or (num_arrows not in ARROWS_VALUES)
            or (stamina not in STAMINA_VALUES)
        ):
            raise ValueError

        self.health = enemy_health
        self.arrows = num_arrows
        self.stamina = stamina

    def show(self):
        return (self.health, self.arrows, self.stamina)

    def __str__(self):
        return f"({self.health},{self.arrows},{self.stamina})"


def value_iteration():
    utilities = np.zeros((POSITIONS, MATERIAL_RANGE, ARROWS_FACTOR, HEALTH_RANGE))
    actions = np.zeros((POSITIONS, MATERIAL_RANGE, ARROWS_FACTOR, HEALTH_RANGE))
    iterations = -1
    delta = 100000000
    while delta > DELTA:
        # iteration
        temp = np.zeros((POSITIONS, MATERIAL_RANGE, ARROWS_FACTOR, HEALTH_RANGE))
        for p1 in range(POSITIONS):
            for m1 in range(MATERIAL_RANGE):
                for a1 in range(ARROWS_RANGE):
                    for h1 in range(HEALTH_RANGE):
                        # for each state
                        for a1 in range(NUM_ACTIONS):
                            # for each action
                            # check enemy condition
                            # if enemy attacks no action is done
                            # if ENEMY_STATE == D:
                            #     int num = r
                            pass


def action(action, state, costs):
    state = State(*state)

    if action == "ACTION_SHOOT":
        if state.arrows == 0 or state.stamina == 0:
            return None, None

        new_arrows = state.arrows - 1
        new_stamina = state.stamina - 1

        choices = []
        choices.append(
            (
                0.5,
                State(max(HEALTH_VALUES[0], state.health - 1), new_arrows, new_stamina),
            )
        )
        choices.append((0.5, State(state.health, new_arrows, new_stamina)))

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_SHOOT] + REWARD[choice[1].show()])

        return cost, choices

    elif action == "ACTION_RECHARGE":
        choices = []
        choices.append(
            (
                0.8,
                State(
                    state.health,
                    state.arrows,
                    min(STAMINA_VALUES[-1], state.stamina + 1),
                ),
            )
        )
        choices.append((0.2, State(state.health, state.arrows, state.stamina)))

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_RECHARGE] + REWARD[choice[1].show()])

        return cost, choices

    elif action == "ACTION_DODGE":
        if state.stamina == 0:
            return None, None

        choices = []
        choices.append(
            (
                0.64,
                State(
                    state.health,
                    min(ARROWS_VALUES[-1], state.arrows + 1),
                    max(STAMINA_VALUES[0], state.stamina - 1),
                ),
            )
        )
        choices.append(
            (
                0.16,
                State(
                    state.health,
                    state.arrows,
                    max(STAMINA_VALUES[0], state.stamina - 1),
                ),
            )
        )
        choices.append(
            (
                0.04,
                State(
                    state.health,
                    state.arrows,
                    max(STAMINA_VALUES[0], state.stamina - 2),
                ),
            )
        )
        choices.append(
            (
                0.16,
                State(
                    state.health,
                    min(ARROWS_VALUES[-1], state.arrows + 1),
                    max(STAMINA_VALUES[0], state.stamina - 2),
                ),
            )
        )

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_DODGE] + REWARD[choice[1].show()])

        return cost, choices
