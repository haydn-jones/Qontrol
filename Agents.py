import numpy as np
import gym
from QTable import DiscretizeQTable
from random import random
import math

class CartPoleQontrol():
    """Class that trains a Q-Learning agent to play CartPole
    """
    def __init__(self, lows=[-2.5, -4.5, -0.28, -4.0], highs=[2.5, 4.5, 0.28, 4.0], bin_counts=[2, 3, 3, 6], discount_factor=0.999):
        """
        Notes
        -----
        The default lows and highs were found by running `find_observed_observation_bounds` in
        utils.py for 1,000,000 episodes (or by looking at the cartpole source code). The bin
        counts were found by a gridsearch (not yet, but they will ;) ).

        Parameters
        ----------
        lows : :obj:`list` of :obj:`float`
            Lower bounds of each element in the observation (elements will be clipped to this)
        highs : :obj:`list` of :obj:`float`
            Upper bounds of each element in the observation (elements will be clipped to this)
        bin_counts : :obj:`list` of :obj:`int`
            Number of bins used for each index. 1 bin means the element will not be used as it
            will always be mapped to bin 0
        """
        self.discount_factor = discount_factor

        self.env = gym.make("CartPole-v1")

        self.solved_avg = 195     # Solved when average performance is 195
        self.solved_period = 100  # over 100 episodes
        self.cur_avg = 0          # keeps track of the rolling average

        self.episodes = 0         # How many training episodes have been run

        self.Q = DiscretizeQTable(2, lows, highs, bin_counts)


    def train_episode(self, visualize=False):
        self.episodes += 1
        total_reward = 0

        epsilon = self.get_episilon()
        learning_rate = self.get_learning_rate()

        env = gym.make("CartPole-v1")
        observation = env.reset()
        done = False
        while not done:
            if visualize:
                env.render()

            action = self.get_action(env, observation, epsilon)

            next_observation, reward, done, _ = env.step(action)

            self.update_bellman(observation, action, reward, next_observation, learning_rate)

            observation = next_observation
            total_reward += 1

        env.close()

        return total_reward

    def get_action(self, env, state, epsilon):
        if random() < epsilon:
            return env.action_space.sample()

        return np.argmax(self.Q[state])

    def get_episilon(self):
        return math.pow(math.e, -1 * self.episodes / 100)

    def get_learning_rate(self):
        return 0.1

    def update_bellman(self, state, action, reward, next_state, learning_rate):
        new_q = (1 - learning_rate) * self.Q[state, action] + learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]))

        self.Q[state, action] = new_q