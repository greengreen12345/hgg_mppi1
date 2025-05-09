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

def main():
    args = get_args()
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

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

        tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

