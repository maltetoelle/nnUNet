from typing import List
from typing import Dict
import torch
from torch import autocast
import torch.nn.functional as F
from torch import distributed as dist
import numpy as np
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerPartialLabelledFL(nnUNetTrainer):
        
    def _build_loss(self):
        loss = DC_and_BCE_loss({},
                               {'batch_dice': self.configuration_manager.batch_dice,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                               use_ignore_label=self.label_manager.ignore_label is not None,
                               dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def get_training_transforms(self, *args, **kwargs):
        tr_transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        tr_transforms.transforms.append(self.to_one_hot_transform)
        return tr_transforms
    
    def get_validation_transforms(self, *args, **kwargs):
        val_transforms = nnUNetTrainer.get_validation_transforms(*args, **kwargs)
        val_transforms.transforms.append(self.to_one_hot_transform)
        return val_transforms
    
    def to_one_hot_transform(self, **data_dict):
        n_classes = len(self.label_manager.all_labels)
        data_dict['target'] = [F.one_hot(y[:,0].long(), num_classes=n_classes).movedim(-1,1).float() for y in data_dict['target']]
        return data_dict

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        
        # if not self.label_manager.has_regions:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        print('mean_fg_dice', mean_fg_dice, self.current_epoch)
        print('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        print('val_losses', loss_here, self.current_epoch)