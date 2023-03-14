import typing as tp
import numpy as np
import torch, torchvision
from .. import datasets, training, models


class MaskRCNN_RingOutput(tp.NamedTuple):
    labelmap:np.ndarray

class MaskRCNN_RingDetector(models.UNet):  #TODO: should not inherit from UNet
    def __init__(self, *a, nms:float = 0.7, accumulating=False, **kw):
        super().__init__(*a, **kw)
        self.basemodule   = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=False, box_nms_thresh=nms)
        self.accumulating = accumulating
    
    def forward(self, *x):
        return self.basemodule(*x)
    
    def start_training(self, *a, **kw):
        return super().start_training(
            *a, 
            ds_cls          =   MaskRCNN_RingDataset, 
            task_cls        =   MaskRCNN_RingTask, 
            batch_size      =   2, 
            ds_kwargs       =   {'accumulating':self.accumulating}, 
            **kw
        )

    def process_image(self, imagefile:str, score_threshold=0.5, upscale_result='ignored') -> MaskRCNN_RingOutput :
        x           = datasets.load_image(imagefile, downscale=1)
        with torch.no_grad():
            raw         = self(x[None])[0]
        labelmap    = mrcnn_output_to_labelmap(raw, x.shape[-2:], score_threshold)
        return MaskRCNN_RingOutput(labelmap)
    


class MaskRCNN_RingDataset(datasets.Dataset):
    def __init__(self, *a, accumulating:bool=False, **kw):
        super().__init__(*a, **kw)
        self.accumulating = accumulating
    
    def __getitem__(self, i):
        scale   =   min(self.scales)
        x       =   datasets.load_image( self.images[i],                 downscale=scale ) 
        if not self.accumulating:
            y       =   datasets.load_instanced_annotation( self.targets[i], downscale=scale, force_cpu=True ) 
        else:
            #ordered labels
            from ..INBD import load_annotation_for_inbd
            y       =   load_annotation_for_inbd( self.targets[i], downscale=scale, force_cpu=True ) 
        y       =   torch.as_tensor(y)

        if self.augment:
            x       =   datasets.augment_color_jitter(x)
            k       =   np.random.randint(0,4)
            x       =   torch.rot90(x, k, dims=[-2,-1])
            y       =   torch.rot90(y, k, dims=[-2,-1])
            if np.random.random() < 0.5:
                x       = torch.flip( x, dims=[-2,-1] )
                y       = torch.flip( y, dims=[-2,-1] )
        y       =   labelmap_to_boxes(y, self.accumulating)
        return x,y

        
    def load_and_cache_dataset(self, imagefiles:tp.List[str], targetfiles:tp.List[str]) -> None:
        #NOT loading and caching
        #self._create_cache_dir()
        self.images         =   imagefiles
        self.targets        =   targetfiles

    
    def collate_fn(self, items):
        images      = [img for (img, tgt) in items]
        targets     = [tgt for (img, tgt) in items]
        return images, targets


def labelmap_to_boxes(labelmap:np.ndarray, accumulating:bool=False) -> dict:
    labels      =   sorted(np.unique(labelmap))
    result      =   {
        'boxes' :   [],
        'masks' :   [],
        'labels':   [],
    }
    for l in labels:
        if l < 1:
            continue
        if not accumulating:
            mask             = (labelmap == l)
        else:
            mask             = np.isin(labelmap, np.arange(1,l+1) )
        mask             = torch.as_tensor(mask).byte()
        indices          = np.argwhere(mask)
        y0,x0            = torch.min(indices, axis=1)[0]
        y1,x1            = torch.max(indices, axis=1)[0]+1
        result['masks'] += [mask]
        result['boxes'] += [ torch.as_tensor([x0,y0,x1,y1]).float() ]
        result['labels']+= [ torch.ones(1).long()[0] ]
    result = dict([( k, torch.stack(v) ) for k,v in result.items()])
    return result

def mrcnn_output_to_labelmap(raw_output:dict, shape:tuple, score_threshold=0.5) -> np.ndarray:
    sizes       = torchvision.ops.box_area(raw_output['boxes'])
    order       = reversed(np.argsort(sizes))
    labelmap    = np.zeros(shape, dtype='int16')
    for l,i, in enumerate(order, 1):
        if raw_output['scores'][i] < score_threshold or raw_output['labels'][i] != 1:
            continue
        
        mask            = (raw_output['masks'][i,0] > 0.5)
        labelmap[mask]  = l
    return labelmap


class MaskRCNN_RingTask(training.TrainingTask):
    def training_step(self, batch:tp.Tuple[tp.List[torch.Tensor], tp.List[dict]]):
        x,y      = batch
        x        = [_x.to(self.device) for _x in x]
        y        = [dict([ (k, v.to(self.device)) for k,v in _y.items() ]) for _y in y ]

        lossdict = self.basemodule(x,y)
        loss     = sum([v for k,v in lossdict.items()])
        logs     = dict([(k, float(v)) for k,v in lossdict.items()])
        return loss, logs

