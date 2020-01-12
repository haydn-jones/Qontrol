from Agents import CartPoleQontrol
from gridsearch import evaluate_parameters

parameters = {
	"bin_counts": [3, 3, 4, 8],
	"epsilon_time_constant": 100,
	"lr_time_constant": 100,
}

result = evaluate_parameters(
	agent_class=CartPoleQontrol,
	parameters=parameters,
	trials=10,
	n_episodes=600,
	max_episode_length=200
)

print(result)