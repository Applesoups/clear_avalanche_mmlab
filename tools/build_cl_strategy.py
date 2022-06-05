import copy
import torch
from torch import optim
from mmcls.models import losses
from avalanche.training import supervised
from torch.optim import lr_scheduler
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins.load_best import LoadBestPlugin
from avalanche.training import storage_policy

#from apex import amp
#from apex.parallel import DistributedDataParallel

def Build_cl_strategy(cfg, model, device,eval_plugin,args):
    cl_strategy=copy.deepcopy(cfg.cl_strategy)
    strate=cl_strategy.pop('type')
    strategy_type=getattr(supervised, strate)
    optim_cfg=cl_strategy.pop('optimizer')
    optimizer_type=getattr(optim, optim_cfg.pop('type'))
    optim_cfg['params']=model.parameters()
    optimizer=optimizer_type(**optim_cfg)

    scheduler_cfg=cl_strategy.pop('scheduler')
    scheduler_type=getattr(lr_scheduler,scheduler_cfg.pop('type'))
    scheduler=scheduler_type(optimizer=optimizer,**scheduler_cfg)

    #plugin_list=[LRSchedulerPlugin(scheduler)]
    #plugin_list used in CLEAR demo
    plugin_list=[LRSchedulerPlugin(scheduler),LoadBestPlugin('train_stream')]
    loss_cfg=cl_strategy.pop('loss')
    loss_type=getattr(losses, loss_cfg.pop('type'))
    loss=loss_type(**loss_cfg).to(device)

    # sampling buffer
    if cl_strategy.get('buffer'):
        buffer_cfg = cl_strategy.pop('buffer')
        buffer_type = getattr(storage_policy, buffer_cfg.pop('type'))
        buffer = buffer_type(**buffer_cfg)
        cl_strategy['buffer'] = buffer

    model=model.to(device)

    # if args.distributed:
    #     model, optimizer = amp.initialize(model, optimizer)
    #     #torch.distributed.init_process_group(backend='nccl')
    #     model = DistributedDataParallel(model)

    # strategy=strategy_type(model=model, optimizer=optimizer,criterion=loss, evaluator=eval_plugin,device=device,**cl_strategy)
    return strategy_type(model=model, optimizer=optimizer,criterion=loss, evaluator=eval_plugin,plugins=plugin_list,device=device, distributed=args.distributed, **cl_strategy), model