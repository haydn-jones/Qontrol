from Agents import CartPoleQontrol
from utils import TrainPlotter
from tqdm import trange

def main():
	agent = CartPoleQontrol()
	plotter = TrainPlotter()

	for _ in trange(1_000):
		reward = agent.train_episode(visualize=False)
		plotter.update(agent, reward)

		if agent.solved == True:
			print("Solved!")
			break

if __name__ == "__main__":
	main()