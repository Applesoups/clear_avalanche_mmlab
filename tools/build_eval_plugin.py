import copy
from avalanche.evaluation import metrics
from avalanche import logging
from avalanche.training.plugins import EvaluationPlugin


def Build_eval_plugin(cfg, scenario):
    loggers = []
    Metrics = []
    if cfg.loggers:
        for logger_dict in cfg.loggers:
            logger = getattr(logging, logger_dict.pop('type'))
            if logger == logging.TextLogger:
                loggers.append(logger(open(logger_dict.file, 'a')))
            else:
                if logger == logging.WandBLogger:
                    logger_dict['config'] = copy.deepcopy(cfg)
                loggers.append(logger(**logger_dict))

    if cfg.metrics:
        for metric_dict in cfg.metrics:
            metric = getattr(metrics, metric_dict.pop('type'))
            Metrics.append(metric(**metric_dict))

    return EvaluationPlugin(*Metrics, loggers=loggers, benchmark=scenario)
