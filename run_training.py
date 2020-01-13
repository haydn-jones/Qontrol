from agents import CartPoleQontrol
from utils import TrainPlotter
from tqdm import trange

def main():
	agent = CartPoleQontrol()
	# plotter = TrainPlotter()

	for _ in trange(600):
		reward = agent.train_episode(visualize=False, max_steps=200)
		# plotter.update(agent, reward)

if __name__ == "__main__":
	main()