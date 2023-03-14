import typing as tp

import numpy as np
import torch, torchvision
from ..models       import UNet
from ..training     import TrainingTask
from ..segmentation import SegmentationModel, SegmentationOutput
from ..             import INBD, util, datasets


class CartesianOutput(tp.NamedTuple):
    labelmap:    np.ndarray


class CartesianModel(UNet):
    def __init__(self, segmentationmodel:SegmentationModel, *a, input_size:int=512, **kw):
        super().__init__(*a, **kw)
        INBD.extend_input_layers(self, extra_c = 3)
        INBD.replace_batchnorm_with_instancenorm(self)
        self.segmentationmodel = [segmentationmodel]
        self.input_size        = input_size
    
    def preprocess(self, input_rgb:torch.Tensor, segmentation:SegmentationOutput) -> torch.Tensor:
        channels   = [
            input_rgb,
            (segmentation.background      )[None],
            (segmentation.boundary        )[None],
            (segmentation.center     > 0.0)[None],
        ]
        channels = [torch.as_tensor(c).float() for c in channels]
        channels = [datasets.resize_tensor(c, size=[self.input_size]*2, mode='nearest') for c in channels]
        return torch.cat(channels)

    def process_image(self, x:str, upscale_result:bool=False) -> INBD.INBD_Output:
        assert isinstance(x, str), 'Need Imagefile'
        
        imagefile  = x
        segoutput  = self.segmentationmodel[0].process_image(imagefile, upscale_result=False)
        input_rgb  = self.segmentationmodel[0].load_image(imagefile)
        input_rgb  = torchvision.transforms.ToTensor()(input_rgb)

        labelmap   = self.output_to_ring(segoutput.center[None,None], torch.as_tensor(0.0))
        labelmap   = datasets.resize_tensor(
            labelmap[None].float(), size=[self.input_size]*2, mode='nearest'
        )[0]
        for i in range(100):
            x          = self.preprocess(input_rgb, segoutput)
            with torch.no_grad():
                output = self(x[None]).cpu()

            ring       = self.output_to_ring(output, labelmap)
            labelmap   = labelmap + ring*(labelmap.max()+1)
            if self.stopping_condition(labelmap, segoutput.background):
                break
            segoutput  = SegmentationOutput(*segoutput[:3], center=(labelmap > 0) )
        
        if upscale_result:
            labelmap = datasets.resize_tensor(
                labelmap[None], size=input_rgb.shape[-2:], mode='nearest'
            )[0]
        labelmap = labelmap.numpy().astype('uint8')
        return CartesianOutput(labelmap=labelmap)
    
    @staticmethod
    def stopping_condition(labelmap:torch.Tensor, background:torch.Tensor, threshold:float=0.95) -> bool:
        #stopping criterion: 9x% of image covered either by background or by labelmap
        assert len(labelmap.shape) == len(background.shape) == 2
        background = torch.as_tensor(background)
        background = datasets.resize_tensor(background[None], size=labelmap.shape, mode='nearest')[0]
        coverage   = (labelmap > 0.0) | (background > 0.0)
        return (coverage.float().mean() > threshold)
    
    @staticmethod
    def output_to_ring(output:torch.Tensor, previous_labelmap:torch.Tensor) -> torch.Tensor:
        assert len(output.shape)==4 and output.shape[:2]==(1,1)
        output = np.asarray(output)[0,0]
        output = util.select_largest_connected_component(output > 0.0)
        output = torch.as_tensor(output)
        ring   = output * (previous_labelmap == 0)
        return ring
    
    def start_training(self, *a, **kw):
        return super().start_training(
            *a, ds_cls=CartesianDataset, task_cls=CartesianTask, batch_size=4, **kw
        )
    



class CartesianTask(TrainingTask):
    def training_step(self, batch:tp.Tuple[torch.Tensor, torch.Tensor], device='cuda') -> tp.Tuple[torch.Tensor, tp.Dict]:
        x, y   = batch
        x, y   = x.to(device), y.to(device)

        y_pred = self.basemodule(x)
        loss   = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)

        logs   = {}
        logs['loss'] = float(loss)
        logs['acc']  = ((y_pred > 0).float() == y).float().mean().item()
        return loss, logs



class CartesianDataset(INBD.INBD_Dataset):
    def __init__(self, *a, segmentationmodel:SegmentationModel, input_size:int=512, color_jitter:bool=True, **kw):
        super().__init__(*a, **kw)
        self.input_size     = input_size
        self.color_jitter   = color_jitter
        self.load_and_cache_dataset(segmentationmodel)

    def __getitem__(self, i:int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x,y     = super().__getitem__(i)
        y       = y + 1
        target  = np.isin(x.annotation, np.arange(1, y+1))
        target  = torch.as_tensor(target).float()
        input   = torch.cat([
            x.inputimage,
            torch.as_tensor(x.segmentation.background)[None],
            torch.as_tensor(x.segmentation.boundary)[None],
            torch.as_tensor(np.isin(x.annotation, np.arange(1, y))),  #NOTE: not y+1
        ])

        input   = datasets.resize_tensor(input, size=self.input_size, mode='nearest')
        target  = datasets.resize_tensor(target,size=self.input_size, mode='nearest')

        if self.augment and np.random.random() < 0.5:
            input   = torch.flip(input,  dims=(-1,))
            target  = torch.flip(target, dims=(-1,))
        if self.augment:
            k = np.random.randint(0,4)
            input   = torch.rot90(input,  k, dims=[-2,-1])
            target  = torch.rot90(target, k, dims=[-2,-1])
        if self.augment and self.color_jitter:
            input[:3]   = datasets.augment_color_jitter(input[:3])
        
        return input, target
    
    #restore data loader function
    create_dataloader = datasets.Dataset.create_dataloader




