from agents.QTable import DiscretizeQTable
from random import random
import itertools
import numpy as np
import gym

class QontrolAgent():
	""" Base Qontrol agent class
	"""
	def __init__(self, env, lows, highs, bin_counts, actions, discount_factor, epsilon_time_constant, lr_time_constant):
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

		self.epsilon_time_constant = epsilon_time_constant
		self.lr_time_constant = lr_time_constant

		self.env = gym.make(env)

	@profile
	def train_episode(self, visualize=False, max_steps=np.inf):
		""" Runs an episode and updates at each step with the Bellman equation. Returns cumulative reward """\

		epsilon = self.get_epsilon()
		learning_rate = self.get_learning_rate()

		total_reward = 0
		observation = self.env.reset()
		observation = self.Q.discretize_state(observation)
		for i in itertools.count(start=1): # Infinite for, keeps track of iterations
			if visualize:
				self.env.render()

			action = self.get_action(observation, epsilon)
			next_observation, reward, done, _ = self.env.step(action)
			next_observation = self.Q.discretize_state(next_observation)

			self.update_bellman(observation, action, reward, next_observation, learning_rate)

			observation = next_observation
			total_reward += reward
			if done or i >= max_steps:
				break

		self.episodes += 1

		return total_reward

	def get_action(self, state, epsilon):
		""" Epsilon greedy selection of action given state """

		if random() < epsilon:
			return self.env.action_space.sample()

		return np.argmax(self.Q[state])

	def update_bellman(self, state, action, reward, next_state, learning_rate):
		""" Updates Q table accorind to the bellman equation """

		self.Q[state, action] = (1 - learning_rate) * self.Q[state, action] + learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]))

	def get_epsilon(self):
		""" Returns epsilon to be used at episode self.episodes """

		return np.power(np.e, -1 * self.episodes / self.epsilon_time_constant) # Allow epsilon to fully decay to 0

	def get_learning_rate(self):
		""" Returns learning rate to be used at episode self.episodes """

		return max(0.1, np.power(np.e, -1 * self.episodes / self.lr_time_constant))