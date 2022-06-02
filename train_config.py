from importlib.resources import path
from pathlib import Path
import argparse
from mmcv import Config
import torch.nn
from tools import Build_scenario, Build_model,Build_eval_plugin, Build_cl_strategy
import datetime
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device',help='choose from cuda and cpu',default='cuda')
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    # parser.add_argument('--distributed', default=False, type=bool,
    #                 help='whether to use distributed training')
    parser.add_argument('--eval',default=False,type=bool,help='wether to eval')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    scenario=Build_scenario(cfg)
    model=Build_model(cfg)
    model=model

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed=False
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')

    eval_plugin=Build_eval_plugin(cfg,scenario)
    cl_strategy, model=Build_cl_strategy(cfg, model, args.device,eval_plugin,args)

    
    # print('Starting experiment...')
    # results = []
    # for experience in scenario.train_stream:
    #     # train returns a dictionary which contains all the metric values
    #     res = cl_strategy.train(experience)
    #     print('Training completed')

    #     print('Computing accuracy on the whole test set')
    #     # test also returns a dictionary which contains all the metric values
    #     results.append(cl_strategy.eval(scenario.test_stream))
    EVALUATION_PROTOCOL = cfg.scenario.evaluation_protocol
    curr_time = datetime.datetime.now()
    time_str = curr_time.strftime("%Y_%m_%d_%T")
    ROOT = Path(cfg.save_model.model_root)
    ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT=ROOT / time_str
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print("Starting experiment...")
    results = []

    train_metric = {}
    test_metric = {}
    print("Current protocol : ", EVALUATION_PROTOCOL)
    cur_timestep=0
    for index, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        #res = cl_strategy.train(experience)
        if args.eval == False:
            train_metric[cur_timestep] = cl_strategy.train(experience)
        test_metric[cur_timestep] = cl_strategy.eval(scenario.test_stream)
        if index % cfg.save_model.frequency == 0:
            torch.save(
                model.state_dict(),
                str(MODEL_ROOT / f"model{str(int(index)).zfill(2)}.pth")
            )
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        
        with open("./experiments/{}/metric/train_metric_{}.json".format(time_str, args.cl_strategy.type), "w") as out_file:
                    json.dump(train_metric, out_file, indent=6)
                    
                    
        #results.append(cl_strategy.eval(scenario.test_stream))
        with open("./experiments/{}/metric/test_metric_{}.json".format(time_str, args.cl_strategy.type), "w") as out_file:
                    # convert tensor to string for json dump
                    test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'] = \
                        test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'].numpy().tolist()
                    json.dump(test_metric, out_file, indent=6)
        out_file.close()
        cur_timestep += 1
        
if __name__ == '__main__':
    main()