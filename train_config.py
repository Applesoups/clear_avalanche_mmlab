from importlib.resources import path
from pathlib import Path
import argparse
from mmcv import Config
import torch.nn
from tools import Build_scenario, Build_model, Build_eval_plugin, Build_cl_strategy, compute_clear_metrics
import datetime
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device', help='choose from cuda and cpu', default='cuda')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--distributed', default=False, type=bool,
                        help='whether to use distributed training')
    parser.add_argument('--eval', default=False, type=bool, help='whether to eval')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    EVALUATION_PROTOCOL = cfg.scenario.evaluation_protocol
    curr_time = datetime.datetime.now()
    time_str = curr_time.strftime("%Y_%m_%d_%T")
    ROOT = Path(cfg.work_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT = ROOT / time_str
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for logger in cfg.loggers:
        if logger['type'] == 'TextLogger' and 'file' not in logger:
            logger['file'] = logger.get('file', ROOT / f'{time_str}.log')

    scenario = Build_scenario(cfg)
    model = Build_model(cfg)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    eval_plugin = Build_eval_plugin(cfg, scenario)
    cl_strategy, model = Build_cl_strategy(cfg, model, args.device, eval_plugin, args)

    print("Starting experiment...")

    train_metric = {}
    test_metric = {}
    print("Current protocol : ", EVALUATION_PROTOCOL)
    print('training stream length: ',len(scenario.train_stream))
    for index, experience in enumerate(scenario.train_stream):
        # if index==10:
        #     break
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        # res = cl_strategy.train(experience)
        if args.eval == False:
            train_metric[index] = cl_strategy.train(experience)
        test_metric[index] = cl_strategy.eval(scenario.test_stream)

        if cfg.get('checkpoint_config'):
            if (index + 1) % cfg.checkpoint_config.interval == 0:
                ckpt_path = MODEL_ROOT / f'exp_{index + 1}.pth'
                print('Saving checkpoint to', ckpt_path)
                torch.save(model.state_dict(), ckpt_path)
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        # convert tensor to string for json dump
        # test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'] = \
        #     test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'].numpy().tolist()

    if cfg.dataset_type == 'CLEAR':
        print('Computing CLEAR metrics...')
        clear_metrics = compute_clear_metrics(test_metric, final_domain=True)
        for k, v in clear_metrics.items():
            print(f'{k}: {v}')
        import wandb
        wandb.log(clear_metrics)
        test_metric['clear_metrics'] = clear_metrics

    print('Saving evaluation results...')
    with open(MODEL_ROOT / 'eval.json', 'w') as f:
        json.dump({'train': train_metric, 'test': test_metric}, f, indent=4)


if __name__ == '__main__':
    main()
