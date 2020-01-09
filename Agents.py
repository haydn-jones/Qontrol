import numpy as np
import gym
from QTable import DiscretizeQTable
from random import random

class CartPoleQontrol():
    """Class that trains a Q-Learning agent to play CartPole
    """
    def __init__(self, lows=[-2.5, -4.5, -0.28, -4.0], highs=[2.5, 4.5, 0.28, 4.0], bin_counts=[1, 1, 6, 12], discount_factor=0.999):
        """
        Notes
        -----
        The default lows and highs were found by running `find_observed_observation_bounds` in
        utils.py for 1,000,000 episodes (or by looking at the cartpole source code). The bin
        counts were found by a gridsearch (not yet, but they will ;) ).

        Episode is considered terminated after 200 steps, but that isn't enforeced so we do it in the loop

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

        epsilon = self.get_epsilon()
        learning_rate = self.get_learning_rate()

        observation = self.env.reset()

        for step in range(200): # Episode considered done after 200 steps
            if visualize:
                self.env.render()

            action = self.get_action(observation, epsilon)

            next_observation, reward, done, _ = self.env.step(action)

            self.update_bellman(observation, action, reward, next_observation, learning_rate)

            observation = next_observation

            if done:
                break

        return step + 1

    def get_action(self, state, epsilon):
        if random() < epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.Q[state])

    def get_epsilon(self):
        return max(0.1, np.power(np.e, -1 * self.episodes / 100))

    def get_learning_rate(self):
        return max(0.1, np.power(np.e, -1 * self.episodes / 100))

    def update_bellman(self, state, action, reward, next_state, learning_rate):
        self.Q[state, action] += learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action])