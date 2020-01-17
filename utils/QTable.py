import numpy as np

class DiscretizeQTable():
	"""Q Table that takes a continuous domain and discretizes it by means of bucketing (histogram)
	"""
	def __init__(self, n_actions, lows, highs, bin_counts):
		"""
		Parameters
		----------
		n_actions : int
			Number of actions you can take in the environment
		lows : :obj:`list` of :obj:`float`
			Lower bounds of each element in the observation (elements will be clipped to this)
		highs : :obj:`list` of :obj:`float`
			Upper bounds of each element in the observation (elements will be clipped to this)
		bin_counts : :obj:`list` of :obj:`int`
			Number of bins used for each index. 1 bin means the element will not be used as it
			will always be mapped to bin 0
		"""

		self.lows = np.array(lows)
		self.highs = np.array(highs)
		self.bin_widths = (self.highs - self.lows) / bin_counts
		self.valid_bins = np.array(bin_counts, dtype=np.int32) - 1

		self.Q = np.zeros(tuple(bin_counts) + (n_actions, ))

	def discretize_state(self, state):
		state = np.maximum(state, self.lows) # clip observation to at least self.lows
		state = ((state - self.lows) / self.bin_widths).astype(np.int32) # calcualte which bin each component falls into, astype(int) floors
		state = np.minimum(state, self.valid_bins) # elements sometimes get pushed into a bin that doesnt exist

		return state

	def __getitem__(self, state):
		action = slice(None, None, None) # if action is not provided this slices the whole row at state
		if isinstance(state, tuple):
			state, action = state

		return self.Q[tuple(state)][action]

	def __setitem__(self, state, value):
		action = slice(None, None, None) # if action is not provided this slices the whole row at state
		if isinstance(state, tuple):
			state, action = state

		self.Q[tuple(state)][action] = value