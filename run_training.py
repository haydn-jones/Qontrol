import gym
import math
import matplotlib.pyplot as plt
import numpy as np
from Agents import CartPoleQontrol

def main():
	agent = CartPoleQontrol()

	rewards = []
	for _ in range(10_000):
		reward = agent.train_episode(visualize=True)
		rewards.append(reward)

		print(reward)


if __name__ == "__main__":
	main()