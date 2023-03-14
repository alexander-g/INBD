import typing as tp
import numpy as np
import scipy
import skimage


class Boundary(tp.NamedTuple):
    boundarypoints:  np.ndarray
    normals:       np.ndarray
    center:        tp.Tuple[float, float]

    def offset(self, offset:float) -> 'Boundary':
        '''Shift the boundary in or outward'''
        newboundary                 = self._asdict()
        offset                    = np.array(offset).reshape(-1,1)
        newboundary['boundarypoints'] = self.boundarypoints + self.normals*offset
        return Boundary(**newboundary)
    
    def rotate(self, i:int) -> 'Boundary':
        '''Rotate the boundary by `i` number of points'''
        bpoints = np.concatenate([ self.boundarypoints[i:], self.boundarypoints[:i] ])
        normals = np.concatenate([ self.normals[i:],     self.normals[:i] ])
        
        newboundary                 = self._asdict()
        newboundary['boundarypoints'] = bpoints
        newboundary['normals']      = normals
        return Boundary(**newboundary)

    def normalize_rotation(self) -> 'Boundary':
        '''Rotate the points so that the normal angles are strictly increasing'''
        angles      = self.compute_angles()
        i           = np.argmin(angles)
        return self.rotate(i)
    
    def compute_radii(self) -> np.ndarray:
        bpoints     = self.boundarypoints
        center      = self.center
        radii       = np.sqrt( ((bpoints - center)**2).sum(-1) )
        return radii

    def compute_angles(self) -> np.ndarray:
        return np.arctan2( *self.normals.T )
    
    @staticmethod
    def from_polar_coordinates(radii:np.ndarray, angles:np.ndarray, center:tp.Tuple[float,float]) -> 'Boundary':
        normals = np.stack([
            np.sin(angles),
            np.cos(angles),
        ], axis=1)
        bpoints = normals * radii[:,None] + center
        return Boundary(bpoints, normals, center)
    
    @staticmethod
    def from_cartesian_coordinates(points:np.ndarray, center:tp.Tuple[float, float], sort=True) -> 'Boundary':
        dvecs  = (points - center)
        radii  = np.sqrt( (dvecs**2).sum(-1) )
        angles = np.arctan2( *dvecs.T )
        if sort:
            order  = np.argsort(angles)
            radii  = radii[order]
            angles = angles[order]
        return Boundary.from_polar_coordinates(radii, angles, center)
    
    def interpolate(self, new_angles:np.ndarray, circular=False) -> 'Boundary':
        '''Interpolate boundary points at new angles'''
        boundary      = self.normalize_rotation()
        angles      = boundary.compute_angles()
        radii       = boundary.compute_radii()
        normals     = boundary.normals
        if circular:
            angles  = np.concatenate([[angles[-1]-np.pi*2], angles,  [angles[0]+np.pi*2]] )
            radii   = np.concatenate([[radii[-1]],          radii,   [radii[0]] ])
            normals = np.concatenate([[normals[-1]],        normals, [normals[0]]])

        new_radii   = np.interp(new_angles, angles, radii)
        return Boundary.from_polar_coordinates(new_radii, new_angles, self.center)

        new_normals = scipy.interpolate.interp1d(
            angles, normals, axis=0, bounds_error=False, fill_value='extrapolate'
        )(new_angles)
        new_normals = new_normals / (new_normals**2).sum(-1, keepdims=True)**0.5
        new_bpoints = new_radii[:,None] * new_normals + boundary.center
        
        return Boundary(new_bpoints, new_normals, boundary.center)

    def resample(self, r_n_ratio:float=5.7) -> 'Boundary':
        '''Resample boundary points to have a fixed euclidean distance to each other across rings (approximately)'''
        angles      = self.compute_angles()
        radii       = self.compute_radii()
        mean_radius = radii.mean()
        new_n       = int( mean_radius * r_n_ratio )
        new_angles  = np.linspace(angles.min(), angles.max(), new_n)
        new_boundary  = self.interpolate(new_angles)
        return new_boundary

    def clip_to_previous_boundary(self, prev_boundary:'Boundary') -> 'Boundary':
        radii0      = self.compute_radii()
        radii1      = prev_boundary.compute_radii()
        new_radii   = np.maximum(radii0, radii1)
        angles      = self.compute_angles()
        
        return Boundary.from_polar_coordinates(new_radii, angles, self.center)
        
    def mask_out_interpolate_boundary(self, mask:np.ndarray, circular=False) -> 'Boundary':
        '''Remove points as indicated in the mask and interpolate them from the remaining'''
        if np.sum(np.asarray(mask)) < 2:
            #fallback if trying to mask out too much
            mask = np.ones_like(mask)
        angles     = self.compute_angles()
        bpoints    = self.boundarypoints
        
        new_boundary = Boundary.from_cartesian_coordinates(
            self.boundarypoints[mask],
            self.center
        )
        new_boundary = new_boundary.interpolate(angles, circular)
        return new_boundary
    
    def slice(self, i0:int, i1:int) -> 'Boundary':
        return Boundary(
            self.boundarypoints[i0:i1],
            self.normals[i0:i1],
            self.center,
        )

    def __len__(self) -> int:
        return len(self.boundarypoints)


def get_accumulated_boundary(labelmap:np.ndarray, label:int, angular_density:float=5.7) -> tp.Union[Boundary, None]:
    '''Compute the boundary accumulated up to the specified label. (Relevant for wedging rings)'''
    center    = np.argwhere(labelmap==1).mean(0)
    acum_mask = np.isin(labelmap, np.arange(1, label+1) )
    boundary  = skimage.segmentation.find_boundaries(acum_mask, mode='outer')
    bpoints   = np.argwhere(boundary)
    if len(bpoints) == 0:
        return None
    return Boundary.from_cartesian_coordinates(bpoints, center).normalize_rotation().resample(angular_density)


def compare_boundaries(boundary0:Boundary, boundary1:Boundary, acc_threshold=4) -> tp.Dict[str, float]:
    boundary1 = boundary1.interpolate(boundary0.compute_angles())
    radii0  = boundary0.compute_radii()
    radii1  = boundary1.compute_radii()
    l1      = np.abs(radii0 - radii1).mean()
    acc     = (np.abs(radii0 - radii1) < acc_threshold).mean()
    return {
        'l1'   : l1,
        'acc'  : acc,
    }

def boundaries_to_svg(boundaries:tp.List[Boundary], viewbox_size:tp.Tuple[int], scale:float=1) -> str:
    #FIXME: scale should be handled in the model output
    import matplotlib.cm as mplcm

    W, H   = viewbox_size
    center = ''
    if len(boundaries) > 0:
        cx,cy  = boundaries[0].center
        center = f'<circle cx="{cx}" cy="{cy}" r="5" />'

    colors   = (mplcm.gist_ncar( np.linspace(0, 1, len(boundaries)) )[:,:3]*255).astype(int)
    polygons = []
    for i,bd in enumerate(reversed(boundaries)):
        bd_points  = bd.boundarypoints * scale
        polypoints = ' '.join([f'{x},{y}' for y,x in bd_points.reshape(-1,2)])
        polygons  += [f'<polygon  points="{polypoints}" fill="rgba({ ",".join( map(str,colors[i]) ) })" stroke="white" />']
    polygons_str = "\n".join(polygons)
    svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        {center}
        {polygons_str}
    </svg>
    '''
    return svg

def boundaries_to_ring_widths(boundaries:tp.List[Boundary], n_angles:int=100, scale:float=1) -> tp.List[np.ndarray]:
    #FIXME: scale should be handled in the model output
    #need common angles first
    angles     = np.linspace(-np.pi, np.pi, n_angles)
    boundaries = [b.interpolate(angles, circular=True) for b in boundaries]
    radii      = [b.compute_radii()*scale              for b in boundaries]
    widths     = []
    for r0, r1 in zip(radii, radii[1:]):
        widths.append( r1 - r0 )
    return widths    

def boundaries_to_ring_width_output(boundaries:tp.List[Boundary], n_angles:int=100, scale:float=1) -> str:
    output      = ''
    ring_widths = boundaries_to_ring_widths(boundaries, n_angles, scale)
    for widths in ring_widths:
        widths_str = [f'{w:.1f}' for w in widths]
        output    += ','.join(widths_str)+'\n'
    return output

