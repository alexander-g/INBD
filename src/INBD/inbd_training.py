import typing as tp
import numpy as np
import torch, torchvision
import PIL.Image

from ..training     import TrainingTask
from ..segmentation import SegmentationModel
from ..             import util
from .inbd_data     import INBD_Dataset, TrainstepData
from .polar_grid    import PolarGrid, estimate_radial_range
from .boundary      import Boundary, get_accumulated_boundary, compare_boundaries

class INBD_Task(TrainingTask):
    def __init__(
        self, 
        *a, 
        labelsmoothing:float        =   0.0, 
        wd_lambda                   =   0.01, 
        per_epoch_it                =   3, 
        bd_augment                  =   True,
        **kw
    ):
        super().__init__(*a, **kw)
        self.labelsmoothing = labelsmoothing
        self.wd_lambda      = wd_lambda
        self.per_epoch_it   = per_epoch_it
        self.bd_augment     = bd_augment
    
    def training_step(self, batch:tp.Tuple[TrainstepData, int], device='cuda') -> tp.Tuple[torch.Tensor, tp.Dict]:
        data, l = batch

        logs:tp.Dict[str, tp.Any]   = {}

        valid_rings                 = np.arange(1, data.annotation.max()+1)
        boundary                    = get_accumulated_boundary(data.annotation[0], l, self.basemodule.angular_density)
        for l in range(l, min(l+self.per_epoch_it, valid_rings.max()) ):
            width     = estimate_radial_range(boundary, data.segmentation.boundary)
            if width is None:
                #fallback
                width     = data.segmentation.boundary.shape[1] / 4
            
            if self.bd_augment:
                boundary  = augment_boundary_offset(boundary) #must come after width estimation
                boundary  = augment_boundary_rotate(boundary)
                boundary  = augment_boundary_jump(boundary)
                width     = augment_width(width)
            
            pgrid     = PolarGrid.construct(data.inputimage, data.segmentation, data.annotation, boundary, width, self.basemodule.concat_radii, device=device)
            
            start_high = False
            if self.basemodule.wd_det is not None:
                w_ytrue    = torch.as_tensor(wedging_ring_target(pgrid, l))[None].float().to(device)
                start_high = w_ytrue[...,0]

            y_pred    = self.basemodule.to(device).forward_from_polar_grid(pgrid, start_high=start_high)
            y_pred, w_ypred    = y_pred['x'], y_pred.get('wd_x')
            y_true    = torch.as_tensor(create_2d_target(pgrid, l+1))[None].to(device).long()
            
            #new boundary
            boundary    = self.basemodule.output_to_boundary(y_pred[0].cpu(), boundary, pgrid)
            
            cse          = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true[None].float()*(1.0-self.labelsmoothing*2)+self.labelsmoothing)
            logs['cse']  = logs.get('cse',  [])    + [ cse ]
            
            offsets       = (y_pred[0,0] > 0).float().argmin(0)
            logs['acc'] = logs.get('acc', [])    + [((offsets - y_true[0].argmin(0)).abs() <= 4).float().mean().item()]


            if w_ypred is not None:
                logs['w_loss']   = logs.get('w_loss', []) + [torch.nn.functional.binary_cross_entropy_with_logits(w_ypred, w_ytrue)*self.wd_lambda]
                logs['w_acc']   = logs.get('w_acc',  []) + [((w_ypred.detach() > 0) == w_ytrue).float().mean().item()]
        
        loss = logs['cse'] + logs.get('w_loss', [])
        loss = torch.stack(loss).mean()
        logs = dict([ (k, np.nanmean( torch.as_tensor(v) )) for k,v in logs.items() ])
        return loss, logs

    def validation_step(self, batch:tp.Tuple[TrainstepData, int], device='cuda') -> tp.Dict:
        self.train() #not using eval mode

        logs:tp.Dict[str, tp.Any] = {}
        data, l                   = batch
        if l != 1:
            #skip all rings after the first one (processing all rings in an image)
            return logs
        
        valid_rings   = np.arange(1, data.annotation.max()+1)
        boundary        = get_accumulated_boundary(data.annotation[0], l, self.basemodule.angular_density)
        for l in range(l, valid_rings.max() ):
            width        = estimate_radial_range(boundary, data.segmentation.boundary)
            if width is None:
                break
            
            pgrid        = PolarGrid.construct(data.inputimage, data.segmentation, data.annotation, boundary, width, self.basemodule.concat_radii, device=device)
            y_pred       = self.basemodule.to(device).forward_from_polar_grid(pgrid)
            y_pred       = y_pred['x']
            boundary     = self.basemodule.output_to_boundary(y_pred[0].cpu(), boundary, pgrid)


            y_true       = torch.as_tensor(create_2d_target(pgrid, l+1))[None].to(device).long()
            ##TODO: compare boundaries directly instead of offsets
            offsets      = (y_pred[0,0] > 0).float().argmin(0)
            logs['v_acc'] = logs.get('v_acc', [])    + [((offsets - y_true[0].argmin(0)).abs() <= 4).float().mean().item()]
        logs = dict([ (k, np.nanmean( torch.as_tensor(v) )) for k,v in logs.items() ])
        return logs

    
    def fit(self, ds_train:INBD_Dataset,  ds_valid:INBD_Dataset=None, **kw):
        #ds_train.load_and_cache_dataset(self.basemodule.segmentationmodel[0])  #cached every epoch in train_one_epoch()
        if ds_valid is not None:
            with torch.autocast('cuda', enabled=self.amp):
                ds_valid.load_and_cache_dataset(self.basemodule.segmentationmodel[0].cuda())
        rc = super().fit(ds_train, ds_valid, **kw)
        self.train() #not using eval mode
        return rc
    
    def train_one_epoch(self, ds_train:INBD_Dataset, *a, **kw):
        with torch.autocast('cuda', enabled=self.amp):
            ds_train.load_and_cache_dataset(self.basemodule.segmentationmodel[0].cuda())
        self.basemodule.segmentationmodel[0].cpu()
        torch.cuda.empty_cache()
        return super().train_one_epoch(ds_train, *a, **kw)




def create_2d_target(pgrid:PolarGrid, ring_i:int) -> np.ndarray:
    this_ring_map = pgrid.annotation[0]
    all_rings_map = np.isin( this_ring_map, np.arange(ring_i+1) )
    return all_rings_map

def wedging_ring_target(pgrid:PolarGrid, ring_i:int, radial_slack=8) -> np.ndarray:
    return (pgrid.annotation[0] == (ring_i+1)).sum(0) > radial_slack



###augmentations
def augment_boundary_rotate(boundary:Boundary) -> Boundary:
    i       = np.random.randint(0, len(boundary.boundarypoints))
    return boundary.rotate(i)

def augment_boundary_offset(boundary:Boundary) -> Boundary:
    n      = len(boundary.boundarypoints)
    offset = np.cos( np.linspace(0, 2*np.pi, n) * np.random.uniform(0.2, 3.0) ) + np.random.uniform(-10, 10)
    return boundary.offset( offset )

def augment_width(width:float) -> float:
    return width * np.random.uniform(0.9, 1.1)

def augment_boundary_jump(boundary:Boundary) -> Boundary:
    n      = len(boundary.boundarypoints)
    offset = np.zeros(n)
    i1     = np.random.randint(1, n//4)
    i0     = np.random.randint(0, n-i1)
    size   = np.random.uniform(10,100)
    dir    = np.random.choice([-1,1])
    offset[i0:][:i1] = size * dir
    return boundary.offset( offset )
