import numpy as np
from QTable import DiscretizeQTable
from itertools import product

class DiscreteQAgentV2():
    def __init__(self, n_actions, lows, highs, bin_counts, discount_factor, learning_rate):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.q_table = DiscretizeQTable(n_actions, lows, highs, bin_counts)

    def evaluate(self, state):
        q_values = self.q_table[state]

        return np.argmax(q_values)
    
    def update_table(self, state, action, reward, next_state):
        new_q = (1 - self.learning_rate) * self.q_table[state, action] + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]))

        self.q_table[state, action] = new_q
