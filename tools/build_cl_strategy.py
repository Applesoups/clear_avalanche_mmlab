import copy
from torch import optim
from mmcls.models import losses
from avalanche.training import supervised

def Build_cl_strategy(cfg, model, device,eval_plugin):
    cl_strategy=copy.deepcopy(cfg.cl_strategy)
    strategy_type=getattr(supervised, cl_strategy.pop('type'))
    
    optim_cfg=cl_strategy.pop('optimizer')
    optimizer_type=getattr(optim, optim_cfg.pop('type'))
    optim_cfg['params']=model.parameters()
    optimizer=optimizer_type(**optim_cfg)
    
    loss_cfg=cl_strategy.pop('loss')
    loss_type=getattr(losses, loss_cfg.pop('type'))
    loss=loss_type(**loss_cfg)
    # strategy=strategy_type(model=model, optimizer=optimizer,criterion=loss, evaluator=eval_plugin,device=device,**cl_strategy)
    
    return strategy_type(model=model, optimizer=optimizer,criterion=loss, evaluator=eval_plugin,device=device,**cl_strategy)