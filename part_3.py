from copy import deepcopy
import os
import json
import cvxpy as cp
import numpy as np

TEAM = 41
Y = [1 / 2, 1, 2]
REWARD = 50
COST = -10 / Y[TEAM % 3]
ATTACK_COST = -40
TASK = 0
GAMMA = 0.999
DELTA = 0.001
HEALTH_RANGE = 5
ARROWS_RANGE = 4
MATERIAL_RANGE = 3
NUM_POSITIONS = 5  # N, S, E, W, C
NUM_STATES = 2

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROW_VALUES = tuple(range(ARROWS_RANGE))
MATERIAL_VALUES = tuple(range(MATERIAL_RANGE))
POSITION_VALUES = tuple(range(NUM_POSITIONS))
STATE_VALUES = tuple(range(NUM_STATES))

HEALTH_FACTOR = 25  # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1  # 0, 1, 2, 3
MATERIAL_FACTOR = 1  # 0, 1, 2

NUM_ACTIONS = 10
ACTION_UP = 0
ACTION_LEFT = 1
ACTION_DOWN = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
ACTION_SHOOT = 5
ACTION_HIT = 6
ACTION_CRAFT = 7
ACTION_GATHER = 8
ACTION_NONE = 9

ACTIONS = {
    0: "UP",
    1: "LEFT",
    2: "DOWN",
    3: "RIGHT",
    4: "STAY",
    5: "SHOOT",
    6: "HIT",
    7: "CRAFT",
    8: "GATHER",
    9: "NONE",
}

POSITIONS = {
    "W": 0,
    "N": 1,
    "E": 2,
    "S": 3,
    "C": 4,
}

PROBS = {
    "enemy": {
        "D": {"D": 0.8, "R": 0.2},
        "R": {"D": 0.5, "R": 0.5},
    },
    "arrows": {
        1: 0.5,
        2: 0.35,
        3: 0.15,
    },
}

ENEMY_STATE = {
    "D": 0,
    "R": 1,
}

REWARDS = {
    "STEP_COST": -10 / Y[TEAM % 3],
    "HIT_REWARD": -40,
    "FINAL_REWARD": 0,
    "STAY": -10 / Y[TEAM % 3],
}


class State:
    def __init__(self, position, arrows, material, enemy_state, health):
        if (
                (position not in POSITION_VALUES)
                or (material not in MATERIAL_VALUES)
                or (arrows not in ARROW_VALUES)
                or (health not in HEALTH_VALUES)
                or (enemy_state not in STATE_VALUES)
        ):
            raise ValueError

        self.position = position
        self.arrows = arrows
        self.material = material
        self.enemy_state = enemy_state
        self.enemy_health = health

    def show(self):
        return (
            self.position,
            self.arrows,
            self.material,
            self.enemy_state,
            self.enemy_health,
        )

    def get_index(self):
        return (
                self.enemy_health
                + HEALTH_RANGE * self.enemy_state
                + HEALTH_RANGE * 2 * self.arrows
                + HEALTH_RANGE * 2 * ARROWS_RANGE * self.material
                + HEALTH_RANGE * 2 * MATERIAL_RANGE * ARROWS_RANGE * self.position
        )

    def __str__(self):
        return f"({self.position}, {self.arrows}, {self.material}, {self.enemy_health})"


def action(action_type, state):
    state = State(*state)

    if state.enemy_health == 0:
        return np.NINF, None

    ###########################################################
    if action_type == ACTION_UP:
        if state.position not in [POSITIONS["S"], POSITIONS["C"]]:
            return None, None

        new_position = (
            POSITIONS["C"] if state.position == POSITIONS["S"] else POSITIONS["N"]
        )

        choices = []
        possibilities = []

        possibilities.append(
            (
                0.85,
                REWARDS["STEP_COST"],
                State(
                    new_position,
                    state.material,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )
        possibilities.append(
            (
                0.15,
                REWARDS["STEP_COST"],
                State(
                    POSITIONS["E"],
                    state.material,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                if state.position in [POSITIONS["C"]]:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                            State(
                                state.position,
                                state.material,
                                0,
                                state.enemy_state,
                                min(
                                    state.enemy_health + 1,
                                    HEALTH_VALUES[HEALTH_RANGE - 1],
                                ),
                            ),
                        )
                    )
                else:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            possibility[1],
                            deepcopy(possibility[2]),
                        )
                    )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_DOWN:
        if state.position not in [POSITIONS["N"], POSITIONS["C"]]:
            return None, None

        new_position = (
            POSITIONS["C"] if state.position == POSITIONS["N"] else POSITIONS["S"]
        )

        choices = []
        possibilities = []

        possibilities.append(
            (
                0.85,
                REWARDS["STEP_COST"],
                State(
                    new_position,
                    state.material,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )
        possibilities.append(
            (
                0.15,
                REWARDS["STEP_COST"],
                State(
                    POSITIONS["E"],
                    state.material,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                if state.position in [POSITIONS["C"]]:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                            State(
                                state.position,
                                state.material,
                                0,
                                state.enemy_state,
                                min(
                                    state.enemy_health + 1,
                                    HEALTH_VALUES[HEALTH_RANGE - 1],
                                ),
                            ),
                        )
                    )
                else:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            possibility[1],
                            deepcopy(possibility[2]),
                        )
                    )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_LEFT:
        if state.position not in [POSITIONS["E"], POSITIONS["C"]]:
            return None, None

        if state.position is POSITIONS["E"] and TASK != 1:
            new_position = POSITIONS["C"]
        else:
            new_position = POSITIONS["W"]

        choices = []
        possibilities = []

        if state.position == POSITIONS["C"]:
            possibilities.append(
                (
                    0.85,
                    REWARDS["STEP_COST"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
            possibilities.append(
                (
                    0.15,
                    REWARDS["STEP_COST"],
                    State(
                        POSITIONS["E"],
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
        elif state.position == POSITIONS["E"]:
            possibilities.append(
                (
                    1.0,
                    REWARDS["STEP_COST"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                if state.position in [POSITIONS["C"], POSITIONS["E"]]:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                            State(
                                state.position,
                                state.material,
                                0,
                                state.enemy_state,
                                min(
                                    state.enemy_health + 1,
                                    HEALTH_VALUES[HEALTH_RANGE - 1],
                                ),
                            ),
                        )
                    )
                else:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            possibility[1],
                            deepcopy(possibility[2]),
                        )
                    )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_RIGHT:
        if state.position not in [POSITIONS["W"], POSITIONS["C"]]:
            return None, None

        new_position = (
            POSITIONS["C"] if state.position == POSITIONS["W"] else POSITIONS["E"]
        )

        choices = []
        possibilities = []

        if state.position == POSITIONS["W"]:
            possibilities.append(
                (
                    1.0,
                    REWARDS["STEP_COST"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
        else:
            possibilities.append(
                (
                    0.85,
                    REWARDS["STEP_COST"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
            possibilities.append(
                (
                    0.15,
                    REWARDS["STEP_COST"],
                    State(
                        POSITIONS["E"],
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                if state.position in [POSITIONS["E"], POSITIONS["C"]]:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                            State(
                                state.position,
                                state.material,
                                0,
                                state.enemy_state,
                                min(
                                    state.enemy_health + 1,
                                    HEALTH_VALUES[HEALTH_RANGE - 1],
                                ),
                            ),
                        )
                    )
                else:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            possibility[1],
                            deepcopy(possibility[2]),
                        )
                    )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_HIT:
        if state.position not in [POSITIONS["E"], POSITIONS["C"]]:
            return None, None

        choices = []

        if state.position in [POSITIONS["E"]]:
            if state.enemy_state in [ENEMY_STATE["D"]]:
                # Stays in D
                # Hit
                choices.append(
                    (
                        0.2 * 0.8,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.8 * 0.8,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Changes to R
                # Hit
                choices.append(
                    (
                        0.2 * 0.2,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            ENEMY_STATE["R"],
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.8 * 0.2,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            ENEMY_STATE["R"],
                            state.enemy_health,
                        ),
                    )
                )
            elif state.enemy_state in [ENEMY_STATE["R"]]:
                # Stays in R 0.5
                # Hit
                choices.append(
                    (
                        0.2 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.8 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Attack and changes to D 0.5
                # Hit for sure
                choices.append(
                    (
                        0.5,
                        REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                        State(
                            state.position,
                            state.material,
                            ARROW_VALUES[0],
                            ENEMY_STATE["D"],
                            min(
                                state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]
                            ),
                        ),
                    )
                )
        elif state.position in [POSITIONS["C"]]:
            if state.enemy_state in [ENEMY_STATE["D"]]:
                # Stays in D
                # Hit
                choices.append(
                    (
                        0.1 * 0.8,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.9 * 0.8,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Changes to R
                # Hit
                choices.append(
                    (
                        0.1 * 0.2,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            ENEMY_STATE["R"],
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.9 * 0.2,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            ENEMY_STATE["R"],
                            state.enemy_health,
                        ),
                    )
                )
            elif state.enemy_state in [ENEMY_STATE["R"]]:
                # Stays in R 0.5
                # Hit
                choices.append(
                    (
                        0.1 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 2) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 2),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.9 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            state.arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Attack and changes to D 0.5
                # Hit for sure
                choices.append(
                    (
                        0.5,
                        REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                        State(
                            state.position,
                            state.material,
                            ARROW_VALUES[0],
                            ENEMY_STATE["D"],
                            min(
                                state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]
                            ),
                        ),
                    )
                )

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_STAY:
        new_position = state.position

        choices = []
        possibilities = []

        if state.position in [POSITIONS["W"], POSITIONS["E"]]:
            possibilities.append(
                (
                    1.0,
                    REWARDS["STAY"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
        else:
            possibilities.append(
                (
                    0.85,
                    REWARDS["STAY"],
                    State(
                        new_position,
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )
            possibilities.append(
                (
                    0.15,
                    REWARDS["STAY"],
                    State(
                        POSITIONS["E"],
                        state.material,
                        state.arrows,
                        state.enemy_state,
                        state.enemy_health,
                    ),
                )
            )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                if state.position in [POSITIONS["E"], POSITIONS["C"]]:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            REWARDS["STAY"] + REWARDS["HIT_REWARD"],
                            State(
                                state.position,
                                state.material,
                                0,
                                state.enemy_state,
                                min(
                                    state.enemy_health + 1,
                                    HEALTH_VALUES[HEALTH_RANGE - 1],
                                ),
                            ),
                        )
                    )
                else:
                    choices.append(
                        (
                            possibility[0] * PROBS["enemy"]["R"]["D"],
                            possibility[1],
                            deepcopy(possibility[2]),
                        )
                    )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_CRAFT:
        if not (state.position in [POSITIONS["N"]] and state.material > 0):
            return None, None

        choices = []
        possibilities = []

        possibilities.append(
            (
                1.0,
                REWARDS["STEP_COST"],
                State(
                    state.position,
                    state.material - 1,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )

        for arrow in PROBS["arrows"]:
            possibilities.append(
                (
                    possibilities[0][0] * PROBS["arrows"][arrow],
                    possibilities[0][1],
                    deepcopy(possibilities[0][2]),
                )
            )
            possibilities[-1][2].arrows = min(ARROWS_RANGE - 1, state.arrows + arrow)

        possibilities = possibilities[1:]

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_GATHER:
        if state.position not in [POSITIONS["S"]]:
            return None, None

        choices = []
        possibilities = []
        possibilities.append(
            (
                0.75,
                REWARDS["STEP_COST"],
                State(
                    state.position,
                    min(state.material + 1, MATERIAL_VALUES[MATERIAL_RANGE - 1]),
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )
        possibilities.append(
            (
                0.25,
                REWARDS["STEP_COST"],
                State(
                    state.position,
                    state.material,
                    state.arrows,
                    state.enemy_state,
                    state.enemy_health,
                ),
            )
        )

        for possibility in possibilities:
            if state.enemy_state == ENEMY_STATE["D"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["D"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
            elif state.enemy_state == ENEMY_STATE["R"]:
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["D"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["D"]
                choices.append(
                    (
                        possibility[0] * PROBS["enemy"]["R"]["R"],
                        possibility[1],
                        deepcopy(possibility[2]),
                    )
                )
                choices[-1][2].enemy_state = ENEMY_STATE["R"]

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    ###########################################################
    elif action_type == ACTION_SHOOT:
        if state.arrows == 0 or (state.position in [POSITIONS["N"], POSITIONS["S"]]):
            return None, None

        new_arrows = state.arrows - 1
        choices = []

        if state.position in [POSITIONS["E"]]:
            if state.enemy_state in [ENEMY_STATE["D"]]:
                # Stays in D
                # Hit
                choices.append(
                    (
                        0.9 * 0.8,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.1 * 0.8,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Changes to R
                # Hit
                choices.append(
                    (
                        0.9 * 0.2,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.1 * 0.2,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            state.enemy_health,
                        ),
                    )
                )
            elif state.enemy_state in [ENEMY_STATE["R"]]:
                # Stays in R 0.5
                # Hit
                choices.append(
                    (
                        0.9 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.1 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Attack and changes to D 0.5
                # Hit for sure
                choices.append(
                    (
                        0.5,
                        REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                        State(
                            state.position,
                            state.material,
                            ARROW_VALUES[0],
                            ENEMY_STATE["D"],
                            min(
                                state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]
                            ),
                        ),
                    )
                )
        elif state.position in [POSITIONS["C"]]:
            if state.enemy_state in [ENEMY_STATE["D"]]:
                # Stays in D
                # Hit
                choices.append(
                    (
                        0.5 * 0.8,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.5 * 0.8,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Changes to R
                # Hit
                choices.append(
                    (
                        0.5 * 0.2,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.5 * 0.2,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            state.enemy_health,
                        ),
                    )
                )
            elif state.enemy_state in [ENEMY_STATE["R"]]:
                # Stays in R 0.5
                # Hit
                choices.append(
                    (
                        0.5 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.5 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Attack and changes to D 0.5
                # Hit for sure
                choices.append(
                    (
                        0.5,
                        REWARDS["STEP_COST"] + REWARDS["HIT_REWARD"],
                        State(
                            state.position,
                            state.material,
                            ARROW_VALUES[0],
                            ENEMY_STATE["D"],
                            min(
                                state.enemy_health + 1, HEALTH_VALUES[HEALTH_RANGE - 1]
                            ),
                        ),
                    )
                )
        elif state.position in [POSITIONS["W"]]:
            if state.enemy_state in [ENEMY_STATE["D"]]:
                # Stays in D
                # Hit
                choices.append(
                    (
                        0.25 * 0.8,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.75 * 0.8,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Changes to R
                # Hit
                choices.append(
                    (
                        0.25 * 0.2,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.75 * 0.2,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["R"],
                            state.enemy_health,
                        ),
                    )
                )
            elif state.enemy_state in [ENEMY_STATE["R"]]:
                # Stays in R 0.5
                # Hit
                choices.append(
                    (
                        0.25 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                # Miss
                choices.append(
                    (
                        0.75 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            state.enemy_state,
                            state.enemy_health,
                        ),
                    )
                )
                # Attack and changes to D 0.5
                # Miss for sure
                choices.append(
                    (
                        0.25 * 0.5,
                        REWARDS["STEP_COST"]
                        + (
                            REWARDS["FINAL_REWARD"]
                            if max(HEALTH_VALUES[0], state.enemy_health - 1) == 0
                            else 0
                        ),
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["D"],
                            max(HEALTH_VALUES[0], state.enemy_health - 1),
                        ),
                    )
                )
                choices.append(
                    (
                        0.75 * 0.5,
                        REWARDS["STEP_COST"],
                        State(
                            state.position,
                            state.material,
                            new_arrows,
                            ENEMY_STATE["D"],
                            state.enemy_health,
                        ),
                    )
                )

        cost = 0
        for choice in choices:
            cost += choice[0] * choice[1]
        return cost, choices

    elif action_type == ACTION_NONE:
        if state.enemy_health == 0:
            return None, None
    return None, None


def show(states, actions):
    positions = {
        0: "W",
        1: "N",
        2: "E",
        3: "S",
        4: "C",
    }
    enemystate = {0: "D", 1: "R"}
    dict = []
    length = len(actions)
    for i in range(length):
        st = State(*states[i])
        dict.append([(positions[st.position], st.material * MATERIAL_FACTOR, st.arrows * ARROWS_FACTOR,
                      enemystate[st.enemy_state], st.enemy_health * HEALTH_FACTOR), ACTIONS[actions[i]]])
    return dict


def linear_programming():
    output = {}

    alpha = np.zeros((600, 1))
    initState = State(POSITIONS["C"], 2, 3, ENEMY_STATE["R"], HEALTH_VALUES[HEALTH_RANGE - 1])
    alpha[initState.get_index()][0] = 1
    R = []
    utilities = np.zeros(
        (NUM_POSITIONS, MATERIAL_RANGE, ARROWS_RANGE, NUM_STATES, HEALTH_RANGE)
    )

    ind = 0
    for state, _ in np.ndenumerate(utilities):
        for act_index in range(NUM_ACTIONS):
            cost, states = action(act_index, state)
            if cost is None:  # action not valid
                continue
            elif cost == np.NINF and act_index == ACTION_NONE:  # health = 0
                R.append(0)
                ind += 1

            elif states is not None:
                R.append(cost)
                ind += 1

    A = np.zeros((600, len(R)), dtype=np.float64)
    R = np.array(R)
    R = R.reshape((len(R), 1))
    ind = 0

    for state, _ in np.ndenumerate(utilities):
        i = State(*state).get_index()
        for act_index in range(NUM_ACTIONS):
            cost, choices = action(act_index, state)
            if cost is None:  # action not valid
                continue
            if cost == np.NINF and act_index == ACTION_NONE:  # health = 0
                A[i][ind] = 1
                ind += 1
            if choices is not None:
                check = False
                for nxt in choices:
                    if nxt[2].show() != state:
                        check = True
                        A[i][ind] += nxt[0]
                        A[nxt[2].get_index()][ind] -= nxt[0]
                if check: ind += 1

    x = cp.Variable(shape=R.shape, name="x")
    R = R.T
    constraints = [cp.matmul(A, x) == alpha, x >= 0]
    objective = cp.Maximize(cp.matmul(R, x))
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    arr = list(x.value)
    values = [float(val) for val in arr]

    ind = 0
    a = []
    b = []
    for state, _ in np.ndenumerate(utilities):
        actions = []
        for act_index in range(NUM_ACTIONS):
            cost, choices = action(act_index, state)
            if cost is None:  # action not valid
                continue
            if cost == np.NINF and act_index == ACTION_NONE:  # health = 0
                actions.append(act_index)
            if choices is not None:
                check = False
                for choice in choices:
                    if choice[2].show() != state:
                        check = True
                if check: actions.append(act_index)
        act_idx = np.argmax(arr[ind: ind + len(actions)])
        best_action = actions[act_idx]
        ind += len(actions)
        b.append(best_action)
        a.append(state)
    policy = show(a, b)
    output["a"]=A.tolist()
    output["r"]=R[0].tolist()
    output["alpha"]=alpha.T[0].tolist()
    output["x"]=values
    output["policy"]=policy
    output["objective"]=float(solution)
    path = "outputs/part_3.json"
    obj = json.dumps(output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj)



linear_programming()
