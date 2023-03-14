from . import models, datasets, training, evaluation

import typing as tp
import numpy as np
import torch, torchvision
import PIL.Image




class SegmentationOutput(tp.NamedTuple):
    background: np.ndarray
    ring:       np.ndarray  #instance-agnostic rings, not used
    boundary:   np.ndarray  #boundary between rings
    center:     np.ndarray  #first ring/pith

CLASSES = dict([ (k,i) for i,k in enumerate(SegmentationOutput._fields) ])


class SegmentationModel(models.UNet):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, out_channels=len(CLASSES), **kwargs)

    def start_training(self, *a, **kw):
        return super().start_training(
            *a, ds_cls=SegmentationDataset, task_cls=SegmentationTask, **kw
        )
    
    def forward(self, *a, return_dict=True, **kw) -> dict:
        y  =  super().forward(*a, **kw)
        if return_dict:
            y = dict( [(cls, y[:,i]) for cls,i in CLASSES.items()] )
        return y
    
    def process_image(self, *args, **kwargs) -> SegmentationOutput:
        output = super().process_image(*args, return_dict=False, **kwargs)
        output = dict( [(cls, output[i]) for cls,i in CLASSES.items()] )
        output = SegmentationOutput(**output)
        return output


class SegmentationTask(training.TrainingTask):
    def training_step(self, batch:tp.Tuple[torch.Tensor, torch.Tensor]):
        x,ytrue = batch
        assert len(ytrue.shape) == 4 and ytrue.shape[1] == len(CLASSES)
        x,ytrue = x.to(self.device), ytrue.to(self.device)
        
        ypred:dict   = self.basemodule(x, sigmoid=False)

        bce_fn  = torch.nn.functional.binary_cross_entropy_with_logits
        bce     = [
            bce_fn(ypred['background'], ytrue[:,CLASSES['background']].float()) * 1/100,
            bce_fn(ypred['center'],     ytrue[:,CLASSES['center']].float())     * 1/10,
        ]
        bce     = torch.stack(bce).sum()

        dice    = dice_loss( 
            torch.sigmoid(ypred['boundary'][:,None]), 
            ytrue[:, CLASSES['boundary']][:,None].float() 
        ).mean()
        
        logs    = {'bce': float(bce), 'dice':float(dice) }
        logs.update( dict([
            (f'iou_{cls}', float(evaluation.mIoU( ypred[cls].cpu() > 0.0, ytrue[:,i].cpu().bool() ))) for cls, i in CLASSES.items()
        ] ) )

        loss    = bce + dice
        return loss, logs


class SegmentationDataset(datasets.Dataset):
    @classmethod
    def load_targetfile(cls, targetfile:str, one_hot=True) -> np.array:
        x = datasets.load_instanced_annotation(targetfile)
        
        y = np.ones(x.shape[:2]) * CLASSES['ring']
        y[ x == -1 ]   = CLASSES['background']
        y[ x ==  0 ]   = CLASSES['boundary']
        y[ x ==  1 ]   = CLASSES['center']
        y              = torchvision.transforms.ToTensor()(y).byte()
        return y

    def _load_cached(self, i:int):
        img, tgt = super()._load_cached(i)
        tgt = torch.as_tensor(tgt)
        tgt = torch.nn.functional.one_hot(tgt[0].long(), len(CLASSES)).permute(2,0,1).bool()
        return img, tgt


def dice_score(ypred, ytrue, eps=1):
    '''Per-image dice score'''
    d = torch.sum(ytrue, dim=[2,3]) + torch.sum(ypred, dim=[2,3]) + eps
    n = 2* torch.sum(ytrue * ypred, dim=[2,3] ) +eps
    return torch.mean(n/d, dim=1)

def dice_loss(ypred, ytrue):
    return 1-dice_score(ypred,ytrue)
