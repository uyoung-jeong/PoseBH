from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class CSSEnableHook(Hook):
    """Initiate Cross-dataset Self Supervision at certain epoch. (use for ViTPose + Proto only)

    Args:
        init_epoch (int): Epoch to initialize.
        cluster_options (list): options to test for the number of clusters
        cluster_iter (int): iterations for clustering.
        beta (int): scaling coefficient for k means
    """

    def __init__(self,
                 start_epoch=-1,
                 **kwargs):
        self.start_epoch = start_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch != self.start_epoch:
            return
        else:
            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            is_already_enabled = model.proto_head.css_enabled
            if is_already_enabled:
                print(f'CSS is already enabled.')
            else:
                model.proto_head.css_enabled = True
                print(f'CSS enabled at {epoch}th epoch')
