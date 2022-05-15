import argparse
from mmcv import Config
import torch.nn
from tools import Build_scenario, Build_model,Build_eval_plugin, Build_cl_strategy

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
    
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))

if __name__ == '__main__':
    main()