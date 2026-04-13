from .flow_matching import (    
    FlowMatching,
    FlowMatchingCfg
)
from typing import Union

SCHEDULERS = {
    "flow_matching": FlowMatching,
}

SchedulerCfg = Union[FlowMatchingCfg]

def get_scheduler(cfg: SchedulerCfg):
    return SCHEDULERS[cfg.name](cfg)
