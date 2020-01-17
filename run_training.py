from agents import AcrobotQontrol, CartPoleQontrol
from utils import TrainPlotter
from tqdm import trange

def main():
	agent = CartPoleQontrol()
	plotter = TrainPlotter()

	visualize_after = 50000 # Start visualizing after N steps
	visualize = False
	for episode in trange(200):
		if episode == visualize_after:
			visualize = True

		reward, epsilon, learning_rate = agent.train_episode(visualize=visualize)
		plotter.update(reward, epsilon, learning_rate)


if __name__ == "__main__":
	main()