from agents.QontrolAgent import QontrolAgent
import numpy as np

class CartPoleQontrol(QontrolAgent):
	"""Class that trains a Q-Learning agent to play CartPole
	"""
	def __init__(self,
				lows=[-2.5, -4.5, -0.28, -4.0],
				highs=[2.5, 4.5, 0.28, 4.0],
				bin_counts=[1, 1, 4, 9],
				discount_factor=0.999,
				epsilon_time_constant=50,
				lr_time_constant=50,
	):
		"""
		Notes
		-----
		The default lows and highs were found by running `find_observed_observation_bounds` in
		utils.py for 1,000,000 episodes (or by looking at the cartpole source code). The bin
		counts and time constants were found by a gridsearch (not yet, but they will ;) ).

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
		super().__init__("CartPole-v1", lows, highs, bin_counts, 2, discount_factor, epsilon_time_constant, lr_time_constant)

	def train_episode(self, visualize=False, max_steps=np.inf):
		total_reward = super().train_episode(visualize=visualize, max_steps=max_steps)

		return total_reward