import numpy as np
from itertools import product
import gym
from tqdm import trange

def run_gridsearch(agent_class, parameters, trials, n_episodes, max_episode_length):
	pass
	
def evaluate_parameters(agent_class, parameters, trials, n_episodes, max_episode_length):
	wma_performance = []
	for _ in trange(trials):
		agent = agent_class(**parameters)

		trial_rewards = []
		for _ in range(n_episodes):
			reward = agent.train_episode(visualize=False, max_steps=max_episode_length)
			trial_rewards.append(reward)

		wma_performance.append(calc_wma_reverse(trial_rewards))

	return wma_performance

def calc_wma_reverse(vals):
	""" Calculates a weighted moving average of vals, higher weights to first elements.
		Thus higher wma's will be assigned to faster-converging models.
	"""
	return np.dot(vals, np.arange(len(vals), 0, -1)) / (len(vals) * (len(vals) + 1) / 2)