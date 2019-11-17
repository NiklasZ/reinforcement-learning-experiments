import numpy as np
from typing import List


# Helpers
def map_action(action) -> str:
    if action == 0:
        return "UP"
    if action == 1:
        return "RIGHT"
    if action == 2:
        return "DOWN"
    if action == 3:
        return "LEFT"


def get_best_action(*, q_table: dict, state: int) -> int:
    return np.argmax(q_table[state])


def get_epsilon_greedy_action_probabilities(*, q_table: dict, state: int, number_of_actions: int, epsilon: float) -> \
List[float]:
    probs = np.zeros(number_of_actions)
    best_action = get_best_action(q_table=q_table, state=state)

    for idx in enumerate(probs):
        if idx == best_action:
            probs[idx] = 1 - epsilon + epsilon / number_of_actions
        else:
            probs[idx] = epsilon / number_of_actions

    return probs


# Epsilon-greedy selector
def choose_action(*, q_table: dict, state: int, number_of_actions: int, epsilon: float) -> int:
    if epsilon < 0 or epsilon > 1:
        raise Exception('Expected epsilon to be within [0,1]. Received: {}'.format(epsilon))

    probs = get_epsilon_greedy_action_probabilities(q_table=q_table, state=state, number_of_actions=number_of_actions,
                                                    epsilon=epsilon)
    choice = np.random.choice(np.arange(number_of_actions), p=probs)
    return choice


# Calculates how much a linearly scaled epsilon should change by
def compute_epsilon_change(*, episode_count: int, starting_epsilon: float, finishing_epsilon: float) -> float:
    return (starting_epsilon - finishing_epsilon) / episode_count
