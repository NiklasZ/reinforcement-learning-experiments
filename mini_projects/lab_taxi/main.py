from agent import Agent
from monitor import interact
import gym

from mini_projects.lab_taxi.monte_carlo_agent import MonteCarloAgent

env = gym.make('Taxi-v3')
episodes = 50000
# 50000 episodes, best average: 8.55
agent = MonteCarloAgent(episode_count=episodes, action_count=env.nA, starting_epsilon=0.5, discount_rate=0.9)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=episodes)

# TODO add some visualisations
# TODO look into how to render environment
