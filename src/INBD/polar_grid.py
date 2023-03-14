import typing as tp, warnings
import numpy as np
import scipy
import torch, torchvision

from ..segmentation import SegmentationOutput
from .boundary import Boundary

class PolarGrid(tp.NamedTuple):
    samplingpoints:    np.ndarray  #cartesian coordinates
    image:             torch.Tensor
    segmentation:      torch.Tensor
    annotation:        tp.Union[torch.Tensor, None]
    radii:             tp.Union[torch.Tensor, None]

    @classmethod
    def construct(
        cls, 
        inputimage:         np.ndarray, 
        segmentation:       SegmentationOutput, 
        annotation:         tp.Union[np.ndarray,None], 
        boundary:           Boundary, 
        width:              float, 
        concat_radii:       bool = False,
        N:                  int  = 256,
        device:             str  = 'cpu',
    ) -> 'PolarGrid':
        samplingpoints = cls.compute_samplingpoints_fixed_width(boundary, width, N).to(device)
        image          = cls.sample_data(inputimage,   samplingpoints, 'bilinear', device)

        segmentation   = torch.stack([
            torch.as_tensor(segmentation.boundary),
            torch.as_tensor(segmentation.background),
        ])
        segmentation   = cls.sample_data(segmentation, samplingpoints, 'bilinear', device)
        
        if annotation is not None:
            annotation     = cls.sample_data(annotation,   samplingpoints.cpu(), 'nearest', 'cpu')
        
        radii = None
        if concat_radii:
            radii          = radii_for_polar_grid(samplingpoints, boundary)
            radii          = torch.as_tensor(radii)[None].float()
        samplingpoints = np.asarray(samplingpoints.cpu())
        return PolarGrid(samplingpoints, image, segmentation, annotation, radii)

    @staticmethod
    def compute_samplingpoints_fixed_width(boundary:Boundary, width:float, N=256) -> torch.Tensor:
        startpoints    = boundary.boundarypoints
        normals        = boundary.normals
        samplingpoints = np.stack( [ 
            np.linspace(p, p+n*width, N) for p,n in zip(startpoints, normals)
        ], axis=1 )
        samplingpoints = torch.as_tensor(samplingpoints)
        return samplingpoints
    
    @staticmethod
    def sample_data(data:torch.Tensor, samplingpoints:torch.Tensor, mode:tp.Union['nearest', 'bilinear'], device='cpu') -> torch.Tensor:
        assert len(data.shape)==3
        HW             = torch.as_tensor(data.shape[-2:]).to(samplingpoints.device)
        N,M            = samplingpoints.shape[:2]
        samplingpoints = samplingpoints.reshape(-1,2) / HW *2 - 1 
        samplingpoints = torch.flip( samplingpoints, (-1,) ).float()[None,None].to(device)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') #pytorch is too noisy
            data           = torch.as_tensor(data, device=device)
            pgrid          = torch.nn.functional.grid_sample( data[None].float(), samplingpoints, mode, padding_mode='border' )
            pgrid          = pgrid.reshape( data.shape[0], N, M )
        return pgrid



def estimate_distances_to_next_boundary(boundary:Boundary, segmentation:np.ndarray, slack=8, sort=False) -> np.ndarray:
    assert len(segmentation.shape) == 2
    all_points = np.argwhere(np.asarray(segmentation > 0))
    points     = all_points - boundary.center
    p_angles   = np.arctan2( *points.T )
    p_radii    = (points**2).sum(-1)**0.5
    b_angles   = boundary.compute_angles()
    b_radii    = boundary.compute_radii()
    
    bins = (b_angles[:-1] + b_angles[1:])/2
    ixs  = np.digitize(p_angles, bins)
    if sort:
        #mostly for debugging/visualization
        order   = np.argsort(ixs)
        p_radii = p_radii[order]
        ixs     = ixs[order]
    b_radii = b_radii[ixs]
    ok_mask = (p_radii > b_radii + slack)
    p_radii = np.where( ok_mask , p_radii - b_radii , np.inf)
    bins    = np.arange(ixs.max()+1)
    minima  = scipy.ndimage.minimum(p_radii, ixs, bins)
    minima  = np.where( np.isin(bins, ixs), minima, np.inf )
    return minima

def estimate_radial_range(boundary:Boundary, segmentation:np.ndarray, **kw) -> tp.Union[float, None]:
    '''Estimate how far to sample along the radial dimension'''
    distances  = estimate_distances_to_next_boundary(boundary, segmentation, **kw)
    finitemask = np.isfinite(distances)
    if np.mean(finitemask) < 0.05:
        return None
    distances  = distances[finitemask]
    radial_range = np.percentile(distances, 95) * 1.5
    return radial_range
#legacy typo
estimage_radial_range = estimate_radial_range

def radii_for_polar_grid(samplingpoints:torch.Tensor, boundary:Boundary) -> torch.Tensor:
    center         = torch.as_tensor(boundary.center).to(samplingpoints.device)
    spoints_radii  = ((samplingpoints - center)**2).sum(-1)**0.5
    spoints_radii -= spoints_radii.min()
    spoints_radii /= spoints_radii.max()
    return spoints_radii
