import numpy as np
import gym
from QTable import DiscretizeQTable
from random import random

class QontrolAgent():
    """ Base Qontrol agent class
    """ 
    def __init__(self, env, lows, highs, bin_counts, actions, discount_factor):
        """
        Parameters
        ----------
        lows : :obj:`list` of :obj:`float`
            Lower bounds of each element in the observation (elements will be clipped to this)
        highs : :obj:`list` of :obj:`float`
            Upper bounds of each element in the observation (elements will be clipped to this)
        bin_counts : :obj:`list` of :obj:`int`
            Number of bins used for each index. 1 bin means the element will not be used as it
            will always be mapped to bin 0
        actions : :obj: `int`
            Number of actions you can take in the environment
        discount_factor : :obj: 'float'
            Value to discount future reward by
        """

        self.Q = DiscretizeQTable(actions, lows, highs, bin_counts)
        
        self.discount_factor = discount_factor
        self.episodes = 0 # How many training episodes have been run
        
        self.epsilon_time_constant = 100
        self.lr_time_constant = 100

        self.env = gym.make(env)

    def train_episode(self, visualize=False):
        epsilon = self.get_epsilon()
        learning_rate = self.get_learning_rate()

        total_reward = 0
        done = False
        observation = self.env.reset()
        while not done:
            if visualize:
                self.env.render()

            action = self.get_action(observation, epsilon)
            next_observation, reward, done, _ = self.env.step(action)

            self.update_bellman(observation, action, reward, next_observation, learning_rate)

            observation = next_observation

            total_reward += reward

        self.episodes += 1

        return total_reward

    def get_action(self, state, epsilon):
        if random() < epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.Q[state])

    def update_bellman(self, state, action, reward, next_state, learning_rate):
        self.Q[state, action] += learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action])
    
    def get_epsilon(self):
        return np.power(np.e, -1 * self.episodes / self.epsilon_time_constant) # Allow epsilon to fully decay to 0

    def get_learning_rate(self):
        return max(0.1, np.power(np.e, -1 * self.episodes / self.lr_time_constant))

class CartPoleQontrol(QontrolAgent):
    """Class that trains a Q-Learning agent to play CartPole
    """
    def __init__(self, lows=[-2.5, -4.5, -0.28, -4.0], highs=[2.5, 4.5, 0.28, 4.0], bin_counts=[3, 3, 4, 8], discount_factor=0.999):
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
        discount_factor : :obj: 'float'
            Value to discount future reward by
        """
        super().__init__("CartPole-v1", lows, highs, bin_counts, 2, discount_factor)

    def train_episode(self, visualize=False):
        total_reward = super().train_episode(visualize=visualize)
        
        return total_reward