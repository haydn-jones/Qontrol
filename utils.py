import gym
import numpy as np
from tqdm import trange
import pyqtgraph as pg

class TrainPlotter():
    def __init__(self):
        self.rewards = []
        self.epsilons = []
        self.learning_rates = []

        self.win = pg.GraphicsWindow(title="Run Plot")

        self.reward_plot = self.win.addPlot(row=0, col=0, name="Reward", title="Reward Plot")
        self.reward_curve = self.reward_plot.plot()

        self.e_lr_plot = self.win.addPlot(row=1, col=0, name="e_lr", title="Epsilon / Learning Rate Plot")
        self.e_lr_plot.addLegend()

        self.e_curve = self.e_lr_plot.plot(name="Epsilon", pen='y')
        self.l_curve = self.e_lr_plot.plot(name="Learning rate", pen='r')

        self.e_lr_plot.setXLink('Reward')
        self.e_lr_plot.setYRange(0, 1)

    def update(self, agent, reward):
        self.rewards.append(reward)
        self.epsilons.append(agent.get_epsilon())
        self.learning_rates.append(agent.get_learning_rate())

        self.reward_curve.setData(self.rewards)
        self.e_curve.setData(self.epsilons)
        self.l_curve.setData(self.learning_rates)

        pg.QtGui.QApplication.processEvents()

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