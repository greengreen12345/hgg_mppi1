# import numpy as np
# import time, hydra
# from common import get_args,experiment_setup
#
# @hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_panda")
#
# if __name__=='__main__':
#
# 	args = get_args()
# 	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
#
# 	#args.logger.summary_init(agent.graph, agent.sess)
#
# 	# Progress info
# 	args.logger.add_item('Epoch')
# 	args.logger.add_item('Cycle')
# 	args.logger.add_item('Episodes@green')
# 	args.logger.add_item('Timesteps')
# 	args.logger.add_item('TimeCost(sec)')
#
# 	# Algorithm info
# 	# for key in agent.train_info.keys():
# 	# 	args.logger.add_item(key, 'scalar')
#
# 	# Test info
# 	# for key in tester.info:
# 	# 	args.logger.add_item(key, 'scalar')
#
# 	#args.logger.summary_setup()
#
#
# 	for epoch in range(args.epochs):
# 		print("epoch", epoch, args.epochs)
# 		for cycle in range(args.cycles):
# 			print("cycle", cycle, args.cycles)
# 			args.logger.tabular_clear()
# 			#args.logger.summary_clear()
# 			start_time = time.time()
#
# 			learner.learn(args, env, env_test, agent, buffer)
# 			# tester.cycle_summary()
#
# 			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
# 			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
# 			args.logger.add_record('Episodes', buffer.counter)
# 			args.logger.add_record('Timesteps', buffer.steps_counter)
# 			args.logger.add_record('TimeCost(sec)', time.time()-start_time)
#
# 			#args.logger.tabular_show(args.tag)
# 			#args.logger.summary_show(buffer.counter)
#
# 		tester.epoch_summary()
#
# 	tester.final_summary()




import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
# from scripts.reactive_tamp import REACTIVE_TAMP
# from scripts.sim import run_sim
from m3p2i_aip.config.config_store import ExampleConfig

import json
import logging
import numpy as np

import torch, hydra, zerorpc

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list) and all(isinstance(i, torch.Tensor) for i in x):
        return [i.tolist() for i in obj]

    return obj

#@hydra.main(version_base=None, config_path="/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/config", config_name="config_panda")
def main():
    args = get_args()

    # print("*******args.goal_based*********", args.goal_based)
    # print("********args.her*********", args.her)

    env, env_test, agent, buffer, learner, tester = experiment_setup(args)


    # learner.reactive_tamp = REACTIVE_TAMP(cfg, env)
    # learner.run_sim = run_sim(cfg, env)
    #
    # explore_goal_trajectory = []

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")

    for epoch in range(args.epochs):
        print("*************************epoch***********************", epoch, args.epochs)
        for cycle in range(args.cycles):
            print("*********************************cycle*******************************", cycle, args.cycles)
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, buffer, planner)

            log_entry = {
                "epoch": epoch,
                "cycle": cycle,
                "initial_goals": convert_ndarray(learner.initial_goals),
                                                                         "desired_goals": convert_ndarray(
                    learner.desired_goals),
                "explore_goals": convert_ndarray(learner.explore_goals),
                "trajectories": convert_ndarray(learner.achieved_trajectories),
                "episode_return": convert_ndarray(learner.episode_return),
            }
            with open("explore_goals17.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            # with open("explore_goals.txt", "a") as f:  # "a" 表示追加写入
            #     f.write(f"Epoch {epoch}, Cycle {cycle}: {learner.explore_goal}\n")
            # with open("achieved_trajectories.txt", "a") as f:  # "a" 表示追加写入
            #     f.write(f"Epoch {epoch}, Cycle {cycle}: {learner.achieved_trajectories}\n")
            # print("*********-------")
            # with open("rewards.txt", "a") as f:  # "a" 表示追加写入
            #     f.write(f"Epoch {epoch}, Cycle {cycle}: {learner.achieved_rewards}\n")
            # explore_goal_trajectory.append(learner.explore_goal)
            # print("*************************explore_goal_trajectory********************",explore_goal_trajectory)
            # args.logger.add_record('Epoch', f"{epoch}/{args.epochs}")
            # args.logger.add_record('Cycle', f"{cycle}/{args.cycles}")
            # args.logger.add_record('Episodes', buffer.counter)
            # args.logger.add_record('Timesteps', buffer.steps_counter)
            # args.logger.add_record('TimeCost(sec)', time.time() - start_time)

        tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

