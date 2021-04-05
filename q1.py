import numpy as np
from copy import deepcopy
from functools import reduce
from operator import add
import os

TEAM = 41
Y = [1 / 2, 1, 2]
REWARD = 50
COST = -10 / Y[TEAM % 3]
ATTACK_COST = -40

GAMMA = 0.999
DELTA = 0.001

HEALTH_RANGE = 5
ARROWS_RANGE = 4
MATERIAL_RANGE = 3
NUM_POSITIONS = 5  # n, s, e, w, c
NUM_STATES = 2

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROW_VALUES = tuple(range(ARROWS_RANGE))
MATERIAL_VALUES = tuple(range(MATERIAL_RANGE))

HEALTH_FACTOR = 25  # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1  # 0, 1, 2, 3
MATERIAL_FACTOR = 1  # 0, 1, 2

NUM_ACTIONS = 10

POSITIONS = {
    'N': 0,
    'S': 1,
    'E': 2,
    'W': 3,
    'C': 4,
}

PROBS = {
    "enemy": {
        "D": {"D": 0.8, "R": 0.2},
        "R": {"D": 0.5, "R": 0.5},
    },
    "player": {

    }
}

ENEMY_STATE = {
    'D': 0,
    'R': 1,
}


class State:
    def __init__(self, position, arrows, material, enemy_state, health):
        if (position not in POSITIONS) or (arrows not in ARROW_VALUES) or (material not in MATERIAL_VALUES) or (
                health not in HEALTH_VALUES) or (enemy_state not in ENEMY_STATE):
            raise ValueError

        self.position = position
        self.arrows = arrows
        self.material = material
        self.enemy_state = enemy_state
        self.enemy_health = health

    def show(self):
        return (self.position, self.arrows, self.material, self.enemy_state, self.enemy_health)

    def __str__(self):
        return f'({self.position},{self.arrows},{self.material},{self.enemy_health})'


def action(action_type, state, costs):
    state = State(*state)

    if action_type == 'ACTION_UP':
        if state.position not in ["S", "C"]:
            return None, None
        new_position = "C" if state.position == "S" else "N"
        choices = []

        possibilities = []

        possibilities.append((0.85, State(new_position, state.arrows, state.material, state.enemy_state, state.enemy_health)))
        possibilities.append((0.15, State("E", state.arrows, state.material, state.enemy_state, state.enemy_health)))

        # for new_enemy_state in ["D", "R"]:
        for possibility in possibilities:
            if state.enemy_state == "D":
                choices.append((possibility[0] * PROBS["enemy"]["D"]["R"], possibility[1]))
                choices[-1][1].enemy_state = "R"
                choices.append((possibility[0] * PROBS["enemy"]["D"]["D"], possibility[1]))
                choices[-1][1].enemy_state = "D"
            elif state.enemy_state == "R":
                if state.position in ["E", "C"]:
                    choices.append((possibility[0] * PROBS["enemy"]["R"]["D"],
                                    State(state.position, 0, state.material, state.enemy_state, min(state.enemy_health + 1,  HEALTH_VALUES[HEALTH_RANGE - 1]))))
                else:
                    choices.append((possibility[0] * PROBS["enemy"]["R"]["D"],
                                    possibility[1]))
                choices[-1][1].enemy_state = "D"
                choices.append((possibility[0] * PROBS["enemy"]["R"]["R"], possibility[1]))
                choices[-1][1].enemy_state = "R"

        cost = 0
        for choice in choices:
            # need to calculate total expected cost
            cost += choice[0] * costs[0]
        return cost, choices

    elif action_type == 'ACTION_DOWN':
        if state.position not in ["N", "C"]:
            return None, None

    elif action_type == 'ACTION_LEFT':
        if state.position not in ["E", "C"]:
            return None, None
    elif action_type == 'ACTION_RIGHT':
        if state.position not in ["W", "C"]:
            return None, None
    elif action_type == 'ACTION_HIT':
        if state.position not in ["E", "C"]:
            return None, None
    elif action_type == 'ACTION_STAY':
        pass
    elif action_type == 'ACTION_CRAFT':
        if state.position in ["N"] and state.arrows < ARROW_VALUES[ARROWS_RANGE - 1]:
            pass
        else:
            return None, None
    elif action_type == 'ACTION_GATHER':
        if state.position in ["S"] and state.material < MATERIAL_VALUES[MATERIAL_RANGE - 1]:
            pass
        else:
            return None, None

    # shoot arrow
    elif action_type == 'ACTION_SHOOT':
        if state.arrows == 0 or (state.position in ["N", "S"]):
            return None, None

        new_arrows = state.arrows - 1
        choices = []

        if state.position in ["E"]:
            # 0.9
            if state.enemy_state in ["D"]:
                # stays in D
                # hit
                choices.append((0.9 * 0.8, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append(
                    (0.1 * 0.8,
                     State(state.position, new_arrows, state.material, state.enemy_state, state.enemy_health)))
                # changes to R
                # hit
                choices.append((0.9 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.1 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                 state.enemy_health)))
            elif state.enemy_state in ["R"]:
                # stays in R 0.5
                # hit
                choices.append((0.9 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.1 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 state.enemy_health)))
                # attack and changes to D 0.5
                # hit for sure
                choices.append((0.5, State(state.position, ARROW_VALUES[0], state.material, ENEMY_STATE["D"],
                                           min(state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]))))
        elif state.position in ["C"]:
            # 0.5
            if state.enemy_state in ["D"]:
                # stays in D
                # hit
                choices.append((0.5 * 0.8, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append(
                    (0.5 * 0.8,
                     State(state.position, new_arrows, state.material, state.enemy_state, state.enemy_health)))
                # changes to R
                # hit
                choices.append((0.5 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.5 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                 state.enemy_health)))
            elif state.enemy_state in ["R"]:
                # stays in R 0.5
                # hit
                choices.append((0.5 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.5 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                 state.enemy_health)))
                # attack and changes to D 0.5
                # hit for sure
                choices.append((0.5, State(state.position, ARROW_VALUES[0], state.material, ENEMY_STATE["D"],
                                           min(state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]))))
        elif state.position in ["W"]:
            # 0.25
            if state.enemy_state in ["D"]:
                # stays in D
                # hit
                choices.append((0.25 * 0.8, State(state.position, new_arrows, state.material, state.enemy_state,
                                                  max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append(
                    (0.75 * 0.8,
                     State(state.position, new_arrows, state.material, state.enemy_state, state.enemy_health)))
                # changes to R
                # hit
                choices.append((0.25 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                  max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.75 * 0.2, State(state.position, new_arrows, state.material, ENEMY_STATE["R"],
                                                  state.enemy_health)))
            elif state.enemy_state in ["R"]:
                # stays in R 0.5
                # hit
                choices.append((0.25 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                  max(HEALTH_VALUES[0], state.enemy_health - 1))))
                # miss
                choices.append((0.75 * 0.5, State(state.position, new_arrows, state.material, state.enemy_state,
                                                  state.enemy_health)))
                # attack and changes to D 0.5
                # miss for sure
                choices.append((0.5, State(state.position, ARROW_VALUES[0], state.material, ENEMY_STATE["D"],
                                           state.enemy_health)))

        cost = 0
        for choice in choices:
            # need to calculate total expected cost
            cost += choice[0] * costs[0]

        return cost, choices
    elif action_type == 'ACTION_NONE':
        pass


def value_iteration():
    utilities = np.zeros((POSITIONS, ARROWS_RANGE, MATERIAL_RANGE, NUM_STATES, HEALTH_RANGE))
    policies = np.full((POSITIONS, ARROWS_RANGE, MATERIAL_RANGE, NUM_STATES, HEALTH_RANGE), -1, dtype='int')
    index = 0
    done = False
    while not done:
        # iteration
        temp = np.zeros(utilities.shape)
        delta = np.NINF
