import numpy as np
from collections import defaultdict

from shared_utilities.helpers import compute_epsilon_change, choose_epsilon_greedy_action, get_returns


# Constant-Alpha

class MonteCarloAgent:

    def __init__(self, *, episode_count, action_count, alpha=0.02, starting_epsilon=1, finishing_epsilon=0,
                 discount_rate=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.action_count = action_count
        self.Q = defaultdict(lambda: np.zeros(action_count))
        self.epsilon_change = compute_epsilon_change(episode_count=episode_count, starting_epsilon=starting_epsilon,
                                                     finishing_epsilon=finishing_epsilon)
        self.current_epsilon = starting_epsilon
        self.current_episode = []
        self.discount_rate = discount_rate
        self.alpha = alpha

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return choose_epsilon_greedy_action(q_table=self.Q, state=state, number_of_actions=self.action_count,
                                            epsilon=self.current_epsilon)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.current_episode.append((state, action, reward))

        if done:
            self.update_estimates()
            self.current_episode = []
            self.current_epsilon -= self.epsilon_change

    def update_estimates(self):
        returns = get_returns(episode=self.current_episode, discount_rate=self.discount_rate)

        for r, event in zip(returns, self.current_episode):
            state, action, *rest = event
            self.Q[state][action] = calculate_update(estimate=self.Q[state][action], visit_return=r, alpha=self.alpha)


def calculate_update(*, estimate, visit_return, alpha):
    return estimate + alpha * (visit_return - estimate)
