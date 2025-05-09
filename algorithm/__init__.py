from .ddpg import DDPG
from skrl.utils.model_instantiators.torch import deterministic_model
import gymnasium
from skrl.memories.torch import RandomMemory

def create_agent(args):
	# spaces
	observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
	#action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(9,))
	action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
	device = "cuda:0"
	# models
	network = [
		{
			"name": "net",
			"input": "STATES",
			"layers": [64, 64],
			"activations": "elu",
		}
	]
	models = {}
	models["policy"] = deterministic_model(
		observation_space=observation_space,
		action_space=action_space,
		device=device,
		network=network,
		output="ACTIONS",
	)
	models["target_policy"] = deterministic_model(
		observation_space=observation_space,
		action_space=action_space,
		device=device,
		network=network,
		output="ACTIONS",
	)
	models["critic"] = deterministic_model(
		observation_space=observation_space,
		action_space=action_space,
		device=device,
		network=network,
		output="ONE",
	)
	models["target_critic"] = deterministic_model(
		observation_space=observation_space,
		action_space=action_space,
		device=device,
		network=network,
		output="ONE",
	)
	memory = RandomMemory(memory_size=50, num_envs=1, device=device)

	return DDPG(
		models=models,
		memory=memory,
		observation_space=observation_space,
		action_space=action_space,
		device="cuda:0",
		cfg=args,
	)