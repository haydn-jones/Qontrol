import gym
import numpy as np
from tqdm import trange

def find_observed_observation_bounds(env, episodes=1_000_000):
    """ Finds the OBSERVED max and min values of an environments observations by randomly inputting for `episodes` episodes """

    env = gym.make(env)
    min_ = np.array([np.inf] * env.observation_space.shape[0])
    max_ = np.array([-np.inf] * env.observation_space.shape[0])

    for _ in trange(episodes):
        env.reset()
        done = False
        while done == False:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

            min_ = np.minimum(observation, min_) # Element-wise
            max_ = np.maximum(observation, max_) # Element-wise

            if done:
                break

    env.close()

    return min_, max_