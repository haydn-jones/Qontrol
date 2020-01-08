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
        self.q_table = {}

        self.n_actions = n_actions
        
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.bin_counts = np.array(bin_counts, dtype=int)
        
        self.bin_widths = (self.highs - self.lows) / self.bin_counts

        self.Q = np.zeros(tuple(self.bin_counts) + (self.n_actions, ), dtype=np.float32)


    def discretize_state(self, state):
        state = np.clip(state, self.lows, self.highs) # clip observation to [low, high]
        hist = np.floor((state - self.lows) / self.bin_widths) # calcualte which bin each component falls into
        hist = np.clip(hist, 0, self.bin_counts - 1).astype(int) # Observations at boundaries will be placed in bin outside range

        return hist

    def __getitem__(self, state):
        action = slice(None, None, None) # if action is not provided this slices the whole row at state
        if isinstance(state, tuple):
            state, action = state

        state = self.discretize_state(state)

        return self.Q[tuple(state)][action]

    def __setitem__(self, state, value):
        action = slice(None, None, None) # if action is not provided this slices the whole row at state
        if isinstance(state, tuple):
            state, action = state

        state = self.discretize_state(state)

        self.Q[tuple(state)][action] = value