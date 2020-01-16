from agents import CartPoleQontrol
from utils import TrainPlotter
from tqdm import trange

def main():
	agent = CartPoleQontrol()
	# plotter = TrainPlotter()

	for _ in trange(1000):
		reward, epsilon, learning_rate = agent.train_episode(visualize=False, max_steps=500)
		# plotter.update(reward, epsilon, learning_rate)

	for _ in trange(200):
		reward = agent.evaluate_episode(visualize=False, max_steps=200)
		# plotter.update(reward, 0, 0)


if __name__ == "__main__":
	main()