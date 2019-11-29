import pytest

from shared_utilities.helpers import get_best_action, choose_epsilon_greedy_action, compute_epsilon_change, get_returns


# choose_action
def test_getsOptimalForZeroEpsilon():
    test_data = {0: [-1, 1]}
    assert choose_epsilon_greedy_action(q_table=test_data, state=0, number_of_actions=2, epsilon=0) == 1


# get_best_action
def test_BestActionChoice():
    test_data = {0: [-1, 1]}
    assert get_best_action(q_table=test_data, state=0) == 1


def test_calculationCheck():
    starting_epsilon = 0.5
    finishing_epsilon = 0.4
    episode_count = 5
    assert compute_epsilon_change(episode_count=episode_count, starting_epsilon=starting_epsilon,
                                  finishing_epsilon=finishing_epsilon) == pytest.approx(0.02)


# discount of 1
def test_get_returns():
    episode = [(None, None, 10), (None, None, -2), (None, None, 4)]
    returns = get_returns(episode=episode, discount_rate=1)
    assert returns == [12, 2, 4]


def test_get_returns_with_discount():
    episode = [(None, None, 10), (None, None, -2), (None, None, 4)]
    returns = get_returns(episode=episode, discount_rate=0.9)
    assert returns == [10 - 2 * 0.9 + 4 * 0.9 ** 2, -2 + 4 * 0.9, 4]
