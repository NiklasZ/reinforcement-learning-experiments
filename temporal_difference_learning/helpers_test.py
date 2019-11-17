import unittest

from temporal_difference_learning.helpers import get_best_action, choose_action, compute_epsilon_change


class MyTestCase(unittest.TestCase):
    # choose_action
    def test_getsOptimalForZeroEpsilon(self):
        test_data = {0: [-1, 1]}
        self.assertEqual(choose_action(q_table=test_data, state=0, number_of_actions=2, epsilon=0), 1)

    # get_best_action
    def test_BestActionChoice(self):
        test_data = {0: [-1, 1]}
        self.assertEqual(get_best_action(q_table=test_data, state=0), 1)


class ComputeEpsilonChangeTests(unittest.TestCase):
    def test_calculationCheck(self):
        starting_epsilon = 0.5
        finishing_epsilon = 0.4
        episode_count = 5
        self.assertAlmostEqual(compute_epsilon_change(episode_count=episode_count, starting_epsilon=starting_epsilon,
                                                      finishing_epsilon=finishing_epsilon), 0.02)


if __name__ == '__main__':
    unittest.main()
