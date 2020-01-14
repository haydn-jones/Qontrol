import numpy as np
from itertools import product
import gym
from tqdm import trange
from joblib import Parallel, delayed

def run_gridsearch(agent_class, parameters, trials, n_episodes, max_episode_length=np.inf):
	""" Parameters should a dictionary of the form of the form:
		{
			"bin_counts": List of iterables, each iterable contains possible bin values for respective observation element,
			"epsilon_time_constant": iterable of possible epsilon_time_constant values,
			"lr_time_constant": iterable of possible lr_time_constant values,
		}
	"""
	# Grid individual parameters
	gridded_bins = product(*parameters["bin_counts"])
	gridded_epsilons = product(parameters["epsilon_time_constant"])
	gridded_lr_time_constant = product(parameters["lr_time_constant"])

	# Grid all of the parameters
	grid = [x for x in product(gridded_bins, gridded_epsilons, gridded_lr_time_constant)]
	
	# Construct nice dictionary to pass to evaluate_parameters
	for i in range(len(grid)):
		grid[i] = {
			"bin_counts": grid[i][0],
			"epsilon_time_constant": grid[i][1][0], # Unpack 1 element tuple
			"lr_time_constant": grid[i][2][0],      # Unpack 1 element tuple
		}

	print(f"Running gridsearch over {len(grid)} parameter configurations...")
	results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=True)(
		delayed(evaluate_parameters)(agent_class, params, trials, n_episodes, max_episode_length) for params in grid
	)

	for i in range(len(results)):
		results[i] = (grid[i], results[i])

	return results

def evaluate_parameters(agent_class, parameters, trials, n_episodes, max_episode_length=np.inf):
	""" Evaluates a single set of parameters. Returns the WMA of each trial """
	performance = []
	for _ in range(trials):
		agent = agent_class(**parameters)

		trial_rewards = []
		for _ in range(n_episodes):
			reward = agent.train_episode(visualize=False, max_steps=max_episode_length)
			trial_rewards.append(reward)

		performance.append(np.average(trial_rewards))
		# performance.append(calc_wma_reverse(trial_rewards))

	return performance

def calc_wma_reverse(vals):
	""" Calculates a weighted moving average of vals, higher weights to first elements.
		Thus higher wma's will be assigned to faster-converging models.
	"""
	return np.dot(vals, np.arange(len(vals), 0, -1)) / (len(vals) * (len(vals) + 1) / 2)