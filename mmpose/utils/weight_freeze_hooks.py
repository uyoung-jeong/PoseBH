from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EmbFreezeHook(Hook):
    """Freeze embedding network except the prototypes. (use for ViTPose + Proto only)

    Args:
        freeze_epoch (int): Epoch to freeze.
    """

    def __init__(self,
                 freeze_epoch=-1,
                 **kwargs):
        self.freeze_epoch = freeze_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch == self.freeze_epoch:
            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            
            for param in model.proto_head.deconv_layers.parameters():
                if param.requires_grad == True:
                    param.requires_grad = False

            for param in model.proto_head.neck.parameters():
                if param.requires_grad == True:
                    param.requires_grad = False

            for param in model.proto_head.head.parameters():
                if param.requires_grad == True:
                    param.requires_grad = False

@HOOKS.register_module()
class EmbFreezeThawHook(Hook):
    """Freeze and thaw embedding network except the prototypes. (use for ViTPose + Proto only)

    Args:
        freeze_epoch (int): Epoch to freeze.
    """

    def __init__(self,
                 freeze=[],
                 **kwargs):
        self.is_freeze = len(freeze)>0
        self.freeze_cfg = []
        self.flagged_epochs = []
        if self.is_freeze:
            for e in freeze:
                k,v = e.split(',')
                v = int(v)
                self.freeze_cfg.append([k,v])
                self.flagged_epochs.append(v)
        self.nonlearnable_params = []

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch in self.flagged_epochs:
            frz_cfg_idx = self.flagged_epochs.index(epoch)
            operation = self.freeze_cfg[frz_cfg_idx][0]
            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            
            if operation == 'freeze':
                for pi, param in enumerate(model.proto_head.parameters()):
                    if param.requires_grad == True:
                        param.requires_grad = False
                    elif pi not in self.nonlearnable_params:
                        self.nonlearnable_params.append(pi)
            else:
                for pi, param in enumerate(model.proto_head.parameters()):
                    if pi not in self.nonlearnable_params:
                        param.requires_grad = True
            print(f"prototypes {operation} at {epoch}th epoch.")

@HOOKS.register_module()
class ProtoFreezeHook(Hook):
    """Freeze prototypes after certain epoch. (use for ViTPose + Proto only)

    Args:
        freeze_epoch (int): Epoch to freeze.
    """

    def __init__(self,
                 freeze_epoch=-1,
                 **kwargs):
        self.freeze_epoch = freeze_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch == self.freeze_epoch:
            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            model.proto_head.freeze_weight()

@HOOKS.register_module()
class ProtoFreezeThawHook(Hook):
    """Freeze and thaw prototypes after certain epoch. (use for ViTPose + Proto only)

    Args:
        freeze_epoch (int): Epoch to freeze.
    """

    def __init__(self,
                 freeze=[],
                 **kwargs):
        self.is_freeze = len(freeze)>0
        self.freeze_cfg = []
        self.flagged_epochs = []
        if self.is_freeze:
            for e in freeze:
                k,v = e.split(',')
                v = int(v)
                self.freeze_cfg.append([k,v])
                self.flagged_epochs.append(v)
        
    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch in self.flagged_epochs:
            frz_cfg_idx = self.flagged_epochs.index(epoch)
            operation = self.freeze_cfg[frz_cfg_idx][0]
            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            if operation == 'freeze':
                model.proto_head.freeze_weight()
            else:
                model.proto_head.thaw_weight()
            print(f"prototypes {operation} at {epoch}th epoch.")

@HOOKS.register_module()
class MultiheadFreezeHook(Hook):
    """Freeze multi-head weights after certain epoch. Bakcbone weights are not handled.

    Args:
        freeze_epoch (int): Epoch to freeze. (use for ViTPose + Proto only)
    """

    def __init__(self,
                 head_freeze=[],
                 **kwargs):
        self.is_head_freeze = len(head_freeze)>0
        self.head_freeze_cfg = []
        self.flagged_epochs = []
        if self.is_head_freeze:
            for e in head_freeze:
                k, v = e.split(',')
                v = int(v)
                self.head_freeze_cfg.append([k, v])
                self.flagged_epochs.append(v)
        self.head_nonlearnable_params = []
        self.associate_head_nonlearnable_params = []

    def before_train_epoch(self, runner):
        if not self.is_head_freeze:
            return
        
        epoch = runner.epoch
        if epoch in self.flagged_epochs:
            frz_cfg_idx = self.flagged_epochs.index(epoch)
            operation = self.head_freeze_cfg[frz_cfg_idx][0]

            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            
            if operation == 'freeze':
                for pi, param in enumerate(model.keypoint_head.parameters()):
                    if param.requires_grad == True:
                        param.requires_grad = False
                    elif pi not in self.head_nonlearnable_params:
                        self.head_nonlearnable_params.append(pi)
                if hasattr(model, 'associate_keypoint_heads'):
                    for pi, param in enumerate(model.associate_keypoint_heads.parameters()):
                        if param.requires_grad == True:
                            param.requires_grad = False
                        elif pi not in self.associate_head_nonlearnable_params:
                            self.associate_head_nonlearnable_params.append(pi)
            else:
                for pi, param in enumerate(model.keypoint_head.parameters()):
                    if pi not in self.head_nonlearnable_params:
                        param.requires_grad = True
                if hasattr(model, 'associate_keypoint_heads'):
                    for pi, param in enumerate(model.associate_keypoint_heads.parameters()):
                        if pi not in self.associate_head_nonlearnable_params:
                            param.requires_grad = True
            frozen_params = sum(self.head_nonlearnable_params) + sum(self.associate_head_nonlearnable_params)
            print(f"multi-head {operation} at {epoch}th epoch.")

@HOOKS.register_module()
class BackboneFreezeHook(Hook):
    """Freeze backbone weights after certain epoch.

    Args:
        freeze_epoch (int): Epoch to freeze. (use for ViTPose + Proto only)
    """

    def __init__(self,
                 freeze=[],
                 **kwargs):
        self.is_freeze = len(freeze)>0
        self.freeze_cfg = []
        self.flagged_epochs = []
        if self.is_freeze:
            for e in freeze:
                k, v = e.split(',')
                v = int(v)
                self.freeze_cfg.append([k, v])
                self.flagged_epochs.append(v)
        self.bb_nonlearnable_params = []

    def before_train_epoch(self, runner):
        if not self.is_freeze:
            return
        
        epoch = runner.epoch
        if epoch in self.flagged_epochs:
            frz_cfg_idx = self.flagged_epochs.index(epoch)
            operation = self.freeze_cfg[frz_cfg_idx][0]

            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            
            if operation == 'freeze':
                for pi, param in enumerate(model.backbone.parameters()):
                    if param.requires_grad == True:
                        param.requires_grad = False
                    elif pi not in self.bb_nonlearnable_params:
                        self.bb_nonlearnable_params.append(pi)
            else:
                for pi, param in enumerate(model.backbone.parameters()):
                    if pi not in self.bb_nonlearnable_params:
                        param.requires_grad = True

            frozen_params = sum(self.bb_nonlearnable_params)
            print(f"backbone {operation} at {epoch}th epoch.")

@HOOKS.register_module()
class ACAEFreezeHook(Hook):
    """Freeze ACAE weights after certain epoch.

    Args:
        freeze_epoch (int): Epoch to freeze. (use for ViTPose + Proto only)
    """

    def __init__(self,
                 freeze=[],
                 **kwargs):
        self.is_freeze = len(freeze)>0
        self.freeze_cfg = []
        self.flagged_epochs = []
        if self.is_freeze:
            for e in freeze:
                k, v = e.split(',')
                v = int(v)
                self.freeze_cfg.append([k, v])
                self.flagged_epochs.append(v)
        self.bb_nonlearnable_params = []

    def before_train_epoch(self, runner):
        if not self.is_freeze:
            return
        
        epoch = runner.epoch
        if epoch in self.flagged_epochs:
            frz_cfg_idx = self.flagged_epochs.index(epoch)
            operation = self.freeze_cfg[frz_cfg_idx][0]

            model = runner.model
            if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
                model = model.module
            
            if operation == 'freeze':
                for pi, param in enumerate(model.acae.parameters()):
                    if param.requires_grad == True:
                        param.requires_grad = False
                    else:
                        self.bb_nonlearnable_params.append(pi)

            else:
                for pi, param in enumerate(model.acae.parameters()):
                    if pi not in self.bb_nonlearnable_params:
                        param.requires_grad = True

            frozen_params = sum(self.bb_nonlearnable_params)
            print(f"acae {operation} at {epoch}th epoch. {frozen_params} params already frozen.")

@HOOKS.register_module()
class SharedLayerFreezeHook(Hook):
    """Freeze shared layer weights of the backbone.

    Args:
        freeze_epoch (int): Epoch to freeze. (use for ViTPose + Proto only)
    """

    def __init__(self, epoch,
                 **kwargs):
        self.epoch = epoch
        
    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch != self.epoch:
            return

        model = runner.model
        if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
            model = model.module
        
        n_exp = 0
        n_frozen = 0
        for pi, param_tuple in enumerate(model.backbone.named_parameters()):
            name, param = param_tuple
            if '.experts.' in name:
                n_exp += 1
            if param.requires_grad == True:
                param.requires_grad = False
                n_frozen += 1

        print(f"{n_frozen}/{n_frozen+n_exp} backbone shared layers are frozen at {epoch}th epoch.")
