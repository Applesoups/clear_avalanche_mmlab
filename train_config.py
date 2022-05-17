from importlib.resources import path
import os
import sys
import json
from pathlib import Path
import argparse
from mmcv import Config
import torch.nn
from tools import Build_scenario, Build_model,Build_eval_plugin, Build_cl_strategy
import datetime
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device',help='choose from cuda and cpu',default='cuda')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    scenario=Build_scenario(cfg)
    model=Build_model(cfg)
    model=model.to(args.device)
    eval_plugin=Build_eval_plugin(cfg,scenario)
    cl_strategy=Build_cl_strategy(cfg, model, args.device,eval_plugin)
    
    # print('Starting experiment...')
    # results = []
    # for experience in scenario.train_stream:
    #     # train returns a dictionary which contains all the metric values
    #     res = cl_strategy.train(experience)
    #     print('Training completed')

    #     print('Computing accuracy on the whole test set')
    #     # test also returns a dictionary which contains all the metric values
    #     results.append(cl_strategy.eval(scenario.test_stream))
    # EVALUATION_PROTOCOL = cfg.scenario.evaluation_protocol
    # curr_time = datetime.datetime.now()
    # time_str = curr_time.strftime("%Y_%m_%d_%T")
    # ROOT = Path(cfg.save_model.model_root)
    # ROOT.mkdir(parents=True, exist_ok=True)
    # MODEL_ROOT=ROOT / time_str
    # ROOT.mkdir(parents=True, exist_ok=True)

    print("Starting experiment...")
    results = []
    print("Current protocol : ", EVALUATION_PROTOCOL)
    for index, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = cl_strategy.train(experience)
        if index % cfg.save_model.frequency == 0:
            torch.save(
                model.state_dict(),
                str(MODEL_ROOT / f"model{str(int(index/cfg.save_model.frequency)).zfill(2)}.pth")
            )
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        results.append(cl_strategy.eval(scenario.test_stream))

if __name__ == '__main__':
    main()