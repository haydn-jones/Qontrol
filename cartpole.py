import gym
from DiscreteQAgent import DiscreteQAgentV2
from random import random
import math
import matplotlib.pyplot as plt
import numpy as np

def main():
	agent = DiscreteQAgentV2(2, [-4.8, -5, -0.275, -3.75], [4.8, 5, 0.275, 3.75], [2, 10, 10, 10], 1, 0.1)

	rewards = []
	for i in range(10_000):
		epsilon = math.pow(math.e, -1 * i / 100)
		reward = run_training_episode(agent, epsilon)
		rewards.append(reward)
		print(f"{i}: {reward}\t{epsilon:0.2f}")

def run_training_episode(agent, epsilon):
	env = gym.make("CartPole-v1")

	done = False
	observation = env.reset()
	total_reward = 0

	while not done:
		# env.render()

		if random() < epsilon:
			action = env.action_space.sample()
		else:
			action = agent.evaluate(observation)

		next_observation, reward, done, _ = env.step(action)

		if done:
			if abs(next_observation[2] * 180 / math.pi) > 12:
				reward = -100
			else:
				reward = 100
		
		agent.update_table(observation, action, reward, next_observation)
		observation = next_observation

		total_reward += 1
	
	env.close()

	return total_reward

if __name__ == "__main__":
	main()