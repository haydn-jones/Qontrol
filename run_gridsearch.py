from agents import CartPoleQontrol
from utils import run_gridsearch
import numpy as np

parameters = {
	"bin_counts": [range(1, 10), range(1, 10), range(1, 10), range(1, 10)],
	"epsilon_time_constant": [50, 100, 150, 200],
	"lr_time_constant": [50, 100, 150, 200],
}

results = run_gridsearch(
	agent_class=CartPoleQontrol,
	parameters=parameters,
	trials=20,
	n_episodes=1000,
	max_episode_length=500
)

results = sorted(results, key=lambda x: np.average(x[1]))
for result in results:
	print(result)