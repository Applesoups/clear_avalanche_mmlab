import copy
from torch import optim
from torch.optim import lr_scheduler
from mmcls.models import losses

import extra_strategies
from avalanche.training import supervised
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins.load_best import LoadBestPlugin
from avalanche.training import storage_policy

#from apex import amp
#from apex.parallel import DistributedDataParallel


def Build_cl_strategy(cfg, model, device, eval_plugin, args):
    cl_strategy = copy.deepcopy(cfg.cl_strategy)
    strate = cl_strategy.pop('type')
    try:
        strategy_type = getattr(supervised, strate)
    except AttributeError:
        strategy_type = getattr(extra_strategies, strate)

    plugin_list = []
    if cl_strategy.get('optimizer'):
        optim_cfg = cl_strategy.pop('optimizer')
        optimizer_type = getattr(optim, optim_cfg.pop('type'))
        optim_cfg['params'] = model.parameters()
        optimizer = optimizer_type(**optim_cfg)
        cl_strategy['optimizer'] = optimizer

        if cl_strategy.get('scheduler'):
            scheduler_cfg = cl_strategy.pop('scheduler')
            scheduler_type = getattr(lr_scheduler, scheduler_cfg.pop('type'))
            scheduler = scheduler_type(optimizer=optimizer, **scheduler_cfg)
            plugin_list.append(LRSchedulerPlugin(scheduler))

    # plugin_list used in CLEAR demo
    plugin_list.append(LoadBestPlugin('train_stream'))

    if cl_strategy.get('loss'):
        loss_cfg = cl_strategy.pop('loss')
        loss_type = getattr(losses, loss_cfg.pop('type'))
        loss = loss_type(**loss_cfg).to(device)
        cl_strategy['criterion'] = loss

    # sampling buffer
    if cl_strategy.get('buffer'):
        buffer_cfg = cl_strategy.pop('buffer')
        buffer_type = getattr(storage_policy, buffer_cfg.pop('type'))
        buffer = buffer_type(**buffer_cfg)
        cl_strategy['buffer'] = buffer

    model = model.to(device)

    # if args.distributed:
    #     model, optimizer = amp.initialize(model, optimizer)
    #     #torch.distributed.init_process_group(backend='nccl')
    #     model = DistributedDataParallel(model)

    return strategy_type(
        model=model,
        evaluator=eval_plugin,
        plugins=plugin_list,
        device=device,
        **cl_strategy), model
