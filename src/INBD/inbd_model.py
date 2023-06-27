import typing as tp
import numpy as np
import scipy
import skimage
import PIL.ImageDraw, PIL.Image
import torch, torchvision

from .inbd_training import INBD_Task
from .inbd_data     import INBD_Dataset, TrainstepData
from ..models       import UNet
from ..segmentation import SegmentationModel, SegmentationOutput
from ..             import util
from .boundary      import Boundary, get_accumulated_boundary
from .polar_grid    import PolarGrid, estimate_radial_range



class INBD_Output(tp.NamedTuple):
    labelmap:    np.ndarray
    boundaries:  tp.List['Boundary']
    polar_grids: tp.List['PolarGrid']

class INBD_Model(UNet):
    def __init__(
        self, 
        segmentationmodel:SegmentationModel, 
        *args, 
        wedging_rings         = True,
        angular_density       = 6.28, 
        concat_radii          = False,
        var_ares              = True,
        interpolate_ambiguous = True,
        **kw
    ):
        super().__init__(*args, **kw)
        
        #in list to hide parameters
        self.segmentationmodel = [segmentationmodel]
        self.scale             = segmentationmodel.scale

        self.wd_det = None
        if wedging_rings:
            #wedging ring detection via axial cumsum
            self.wd_det = WedgingRingModule(in_c = 32, deep=True)
                
        #overwrite first conv layer(s) to add more input channels  #TODO: move out into a function
        extra_c = 2 + (1 if concat_radii else 0)
        extend_input_layers(self, extra_c)
        
        #overwrite self.cls to add more input channels
        cls_in_channels = 32+ (1 if wedging_rings else 0)
        self.cls = torch.nn.Conv2d(cls_in_channels, 1, 3, padding=1)
        
        self.var_ares             = var_ares
        self.interpolate_ambiguous = interpolate_ambiguous
        self.concat_radii         = concat_radii
        self.angular_density      = angular_density
        replace_convs_with_circular(self)

        replace_batchnorm_with_instancenorm(self)

    def forward(self, x:torch.Tensor, start_high=False) -> torch.Tensor:
        output = {}

        x = super().forward(x, return_features=True)
        
        if self.wd_det is not None:
            wd_output   = self.wd_det(x, None, start_high)
            output.update(wd_output)
            x           = output['x']
        
        x_cls       = self.cls(x)
        output['x'] = x_cls

        return output

    def forward_from_polar_grid(self, pgrid:'PolarGrid', **kw) -> torch.Tensor:
        x    = pgrid.image
        x    = torch.cat([x, pgrid.segmentation], dim=0)
        if pgrid.radii is not None:
            x    = torch.cat([x, pgrid.radii], dim=0)
        x    = x[None]
        return self(x, **kw)
    
    def output_to_boundary(self, y:torch.Tensor, prev_boundary:Boundary, pgrid:PolarGrid) -> Boundary:
        assert len(y.shape) == 3
        y          = y.cpu().detach()
        #y          = np.asarray(y).argmax(0)                            #channelwise, class (0/1)
        y          = (np.asarray(y)[0] > 0)
        N,M        = y.shape
        offsets0   = y.argmin(0)                                        #first `zero`
        offsets1   = N - y[::-1].argmax(0)                              #last `one`
        
        points     = pgrid.samplingpoints[offsets0, np.arange(M)]
        boundary   = Boundary(points, prev_boundary.normals, prev_boundary.center)

        if self.interpolate_ambiguous:
            offsets_ok = (offsets0==offsets1) | ~y.any(0)                   #~y.any(0): ok if all zero
            boundary     = boundary.mask_out_interpolate_boundary(offsets_ok, circular=True)
        boundary     = boundary.clip_to_previous_boundary(prev_boundary)
        if self.var_ares:
            boundary     = boundary.resample(self.angular_density)
        return boundary
    
    def process_image(self, x:tp.Union[str, SegmentationOutput], max_n=100, upscale_result=False, ) -> INBD_Output:
        if isinstance(x, str):
            imagefile  = x
            output     = self.segmentationmodel[0].process_image(imagefile, upscale_result=False)
            x          = self.segmentationmodel[0].load_image(imagefile)
        elif isinstance(x, SegmentationOutput):
            output     = x
            raise NotImplementedError('Need imagefile')
        
        scale      = self.segmentationmodel[0].scale
        x          = torchvision.transforms.ToTensor()( x )
        x          = torch.nn.functional.interpolate(x[None], scale_factor=1/scale)[0]
        centermask = (output.center > 0)

        all_boundaries = []
        all_pgrids     = []
        boundary       = detected_center_to_boundary( centermask, convex=True, angular_density=self.angular_density ) #smooth, using as starting point
        if boundary is not None:
            all_boundaries = [detected_center_to_boundary( centermask, convex=False, angular_density=None )] #more accurate
            for i in range(max_n):
                width       = estimate_radial_range(boundary, output.boundary)
                if width in [0, None]:
                    break
                pgrid       = PolarGrid.construct(x, output, None, boundary, width, self.concat_radii)
                y_pred      = self.forward_from_polar_grid(pgrid)
                y_pred      = y_pred['x']
                boundary    = self.output_to_boundary(y_pred[0].cpu(), boundary, pgrid)
                all_boundaries.append(boundary)
                all_pgrids.append(pgrid)
        #TODO: scale boundaries
        labelmap = boundaries_to_labelmap(all_boundaries, centermask.shape, filter_threshold=0, scale=1 if not upscale_result else self.scale)
        #NOTE: currently not applying background due to issues in downstream tasks
        #labelmap = apply_background(labelmap, (output.background > 0))
        labelmap = util.filter_labelmap(labelmap, threshold=0.001)
        #after filtering remove corresponding boundaries
        all_boundaries = [all_boundaries[i-1] for i in np.unique(labelmap) if i!=0]
        return INBD_Output(labelmap, all_boundaries, all_pgrids)

    def start_training(self, *a, **kw):
        return super().start_training(*a, ds_cls=INBD_Dataset, task_cls=INBD_Task, **kw)

def boundaries_to_labelmap(boundaries, shape:tuple, scale=1.0, filter_threshold:float=0.001) -> np.ndarray:
    shape = int(shape[0]*scale), int(shape[1]*scale)
    img  = PIL.Image.new('L', shape, 0)
    draw = PIL.ImageDraw.Draw(img)
    for i,b in reversed(list(enumerate(boundaries, 1))):
        draw.polygon( (b.boundarypoints*scale).ravel().tolist(), fill=i )
    labelmap = np.array(img).T
    labelmap = util.filter_labelmap(labelmap, filter_threshold)
    return labelmap

def apply_background(x:np.ndarray, background:np.ndarray):
    if background.shape != x.shape:
        background = skimage.transform.resize(background, x.shape, order=0)
    x = x * ~background.astype(bool)
    return x

def detected_center_to_boundary(centermask:np.ndarray, convex:bool=False, angular_density:float=None) -> tp.Union[None, Boundary]:
    centermask  = util.select_largest_connected_component(centermask > 0)
    if convex:
        centermask  = skimage.morphology.convex_hull_image(centermask)
        return get_accumulated_boundary(centermask, 1, angular_density)
    else:
        contours = skimage.measure.find_contours(centermask)
        if len(contours) == 0:
            return None
        bpoints  = contours[0]
        center   = np.argwhere(centermask).mean(0)
        boundary = Boundary.from_cartesian_coordinates(bpoints, center, sort=False).normalize_rotation()
        if angular_density is not None:
            boundary = boundary.resample(angular_density)
        return boundary




#circular conv
class CircularConv(torch.nn.Conv2d):
    def __init__(self, *a, padding=1, **kw):
        self._padding = padding
        super().__init__(*a, padding=0, **kw)

    def forward(self, x:torch.Tensor, *a, **kw) -> torch.Tensor:
        p0,p1 = self._padding
        assert p0 == p1, (p0,p1)
        p     = p0
        x = torch.cat([x[..., -p:],                   x, x[..., :p]],                    dim=3)
        x = torch.cat([torch.zeros_like(x[...,:p,:]), x, torch.zeros_like(x[...,:p,:])], dim=2)
        return super().forward(x, *a, **kw)

def replace_modules(module:torch.nn.Module, module_factory:tp.Callable, replace_what:type) -> torch.nn.Module:
    for name, child in module.named_children():
        if isinstance(child, replace_what):
            new_child = module_factory(child)
            setattr(module, name, new_child)
        replace_modules(child, module_factory, replace_what)
    return module

def replace_convs_with_circular(module:torch.nn.Module) -> torch.nn.Module:
    def module_factory(module:torch.nn.Conv2d):
        if module.kernel_size in [1, (1,1)]:
            return module
        
        new_module = CircularConv(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                padding  = getattr(module, '_padding', module.padding),
                stride   = module.stride,
                dilation = module.dilation,
                bias     = module.bias is not None,
                groups   = module.groups,
            )
        new_module.load_state_dict(module.state_dict())
        return new_module
    
    return replace_modules(
        module, 
        module_factory,
        replace_what=torch.nn.Conv2d
    )

def replace_batchnorm_with_instancenorm(module:torch.nn.Module) -> torch.nn.Module:
    return replace_modules(
        module,
        module_factory = lambda m: torch.nn.InstanceNorm2d(m.num_features),
        replace_what   = torch.nn.BatchNorm2d,
    )

def extend_input_layers(module:SegmentationModel, extra_c:int) -> torch.nn.Module:
    if module.backbone_name == 'resnet18':
        c1                    = module.backbone.conv1
    elif module.backbone_name == 'mobilenet3l':
        c1                    = module.backbone['0'][0]
    else:
        raise NotImplementedError(module.backbone)
    
    new_c1                = torch.nn.Conv2d(c1.in_channels+extra_c, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=c1.bias is not None, groups=c1.groups)
    new_c1.weight.data[:,:3] = c1.weight.data
    new_c1.weight.data[:,3:] = c1.weight.data[:,:1]
    if module.backbone_name == 'resnet18':
        module.backbone.conv1   = new_c1
    elif module.backbone_name == 'mobilenet3l':
        module.backbone['0'][0] = new_c1

    c1                    = module.up4.conv1x1
    module.up4.conv1x1    = torch.nn.Conv2d(c1.in_channels+extra_c, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=c1.bias is not None, groups=c1.groups)
    module.up4.conv1x1.weight.data[:,:c1.weight.data.shape[1]]    = c1.weight.data




#WedgingRingDetection
class WedgingRingModule(torch.nn.Module):
    def __init__(self, in_c:int, deep=False, pre_sig_offset=0):
        super().__init__()
        self.in_channels = in_c
        if not deep:
            self.proj = torch.nn.Conv2d(in_c, 2, 1, bias=False)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.MaxPool2d( (2,1), stride=(2,1) ),
                torch.nn.Conv2d(in_c, in_c//2, 1),
                torch.nn.InstanceNorm2d(in_c//2),
                torch.nn.ReLU(),

                torch.nn.MaxPool2d( (2,1), stride=(2,1) ),
                torch.nn.Conv2d(in_c//2, in_c//4, 1),
                torch.nn.InstanceNorm2d(in_c//4),
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_c//4, 2, 1, bias=False ),
            )
    
    def forward(self, x:torch.Tensor, x_cls:torch.Tensor, start_high=True) -> tp.Dict[str, torch.Tensor]:
        output = {}

        wd_x    = self.proj(x[:,:self.in_channels])

        wd_x0   = wd_x[:,0].mean(1) 
        wd_x1   = wd_x[:,1].mean(1) 

        wd_x     = (torch.sigmoid(wd_x0) - torch.sigmoid(wd_x1)).cumsum(1) - 15 + start_high*30

        #this will be used for the loss
        output['wd_x'] = wd_x

        #this will be used for further processing
        #subtract maximum
        wd_x    = (wd_x - wd_x.max(1,keepdims=True)[0])
        wd_x    = wd_x.detach()

        #repeat along the radial axis
        wd_x    = torch.repeat_interleave(wd_x[:,None,None], x.shape[2], dim=2 )
        #concatenate as new channel
        x       = torch.cat([x, wd_x], dim=1)

        output['x']     = x
        output['wd_x0'] = wd_x0 #for debugging
        output['wd_x1'] = wd_x1 #for debugging
        return output


