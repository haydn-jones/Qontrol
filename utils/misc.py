import gym
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed


def find_observed_observation_bounds(env, episodes=1_000_000):
	""" Finds the OBSERVED max and min values of an environments observations by randomly inputting for `episodes` episodes """

	results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=True)(
		delayed(run_episode)(env) for i in range(episodes)
	)

	mins = [res[0] for res in results]
	maxs = [res[1] for res in results]

	return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)

def run_episode(env):
	env = gym.make(env)
	min_ = np.array([np.inf] * env.observation_space.shape[0])
	max_ = np.array([-np.inf] * env.observation_space.shape[0])

	env.reset()
	done = False
	while done == False:
		action = env.action_space.sample()
		observation, _, done, _ = env.step(action)

		min_ = np.minimum(observation, min_) # Element-wise
		max_ = np.maximum(observation, max_) # Element-wise

	env.close()

	return (min_, max_)