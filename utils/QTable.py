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
		self.lows = np.array(lows, np.float32)
		self.highs = np.array(highs, np.float32)
		self.bin_counts = np.array(bin_counts, dtype=np.int32)

		self.bin_widths = (self.highs - self.lows) / self.bin_counts

		self.Q = np.zeros(tuple(self.bin_counts) + (n_actions, ), dtype=np.float32)

	def discretize_state(self, state):
		state = np.clip(state, self.lows, self.highs) # clip observation to [low, high]
		hist = ((state - self.lows) / self.bin_widths).astype(np.int32) # calcualte which bin each component falls into, astype(int) floors
		hist = np.minimum(hist, self.bin_counts - 1) # elements sometimes get pushed into a bin that doesnt exist

		return hist

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