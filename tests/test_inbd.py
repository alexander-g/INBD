import tempfile, os
from src import INBD, util, segmentation
import numpy as np
import torch


def test_boundary():
    x      = np.zeros([1000,1000], dtype=int)
    center = [500,500]
    for i in reversed(range(1,5)):
        x[center[0]-i*10:, center[1]-i*10:][:i*20+1, :i*20+1] = i
    
    boundary1 = INBD.get_accumulated_boundary(x, label=1)
    boundary3 = INBD.get_accumulated_boundary(x, label=3)

    assert all(boundary1.center == boundary3.center)
    assert len(boundary1.boundarypoints) < len(boundary3.boundarypoints)


    boundary_rot = boundary1.rotate(1)
    assert len(boundary1.boundarypoints) == len(boundary_rot.boundarypoints)
    assert all(boundary1.boundarypoints[0]  == boundary_rot.boundarypoints[-1])
    assert all(boundary1.boundarypoints[1]  == boundary_rot.boundarypoints[0])

    boundary_norm = boundary_rot.normalize_rotation()
    assert all(boundary_norm.boundarypoints[0] == boundary1.boundarypoints[0])

    boundary_off = boundary1.offset(10)
    assert len(boundary1.boundarypoints) == len(boundary_off.boundarypoints)
    assert abs(sum( (boundary1.boundarypoints[0]  - boundary_off.boundarypoints[0])**2 )**0.5 - 10) < 0.01

    radii  = boundary1.compute_radii()
    assert radii.max() <  15
    assert radii.min() >= 10

    angles = boundary1.compute_angles()
    assert np.all(np.diff(angles) > 0)
    assert -np.pi <= angles.min()
    assert angles.max() <= np.pi

    n = len(boundary1.boundarypoints)
    boundary_x = INBD.Boundary.from_polar_coordinates(np.zeros(n), boundary1.compute_angles(), center)
    assert np.allclose(boundary_x.compute_angles(), boundary1.compute_angles())
    boundary_x = boundary_x.offset( boundary1.compute_radii() )
    assert np.allclose( boundary_x.boundarypoints, boundary1.boundarypoints )
    assert np.allclose( boundary_x.normals,      boundary1.normals )


    boundary_x2 = boundary1.interpolate(new_angles=np.linspace(-np.pi, np.pi, n*2))
    assert len(boundary_x2.boundarypoints) == n*2

    boundary_x3 = boundary_x2.resample()

    #
    x_empty = np.zeros([1000,1000], dtype=int)
    boundary4 = INBD.get_accumulated_boundary(x_empty, label=1)
    assert boundary4 is None



def test_polargrid():
    boundary         = INBD.Boundary.from_polar_coordinates(np.ones(50), np.linspace(-np.pi, np.pi, 50), [0,0])
    samplingpoints = INBD.PolarGrid.compute_samplingpoints_fixed_width(boundary, width=77)
    assert samplingpoints.shape == (256,50,2)

    segoutput = segmentation.SegmentationOutput( *[np.ones([100,100])]*4 )
    pgrid = INBD.PolarGrid.construct(np.ones([3,100,100]), segoutput, np.ones([1,100,100]), boundary, width=50)
    assert (pgrid.image==1).any()
    assert pgrid.image.shape == (3,256,50)
    assert pgrid.annotation.shape

    pgrid = INBD.PolarGrid.construct(np.ones([3,100,100]), segoutput, None, boundary, width=50)
    assert pgrid.annotation is None

def test_estimate_radial_range():
    x      = np.zeros([1000,1000], dtype=int)
    center = [500,500]
    x[center[0]-10:, center[0]-10][:20] = 1
    x[center[0]-10:, center[0]+10][:20] = 1

    x[center[0]-50:, center[0]-50][:100] = 1
    x[center[0]-50:, center[0]+50][:100] = 1
    
    boundary  = INBD.Boundary.from_polar_coordinates(np.ones(50), np.linspace(-np.pi, np.pi, 50), center)
    width   = INBD.estimate_radial_range(boundary, x)
    #assert 0 < width < 20
    assert width > 0

    width   = INBD.estimate_radial_range(boundary, x, slack=30)
    assert width >= 30

    width   = INBD.estimate_radial_range(boundary, x, slack=5000)
    assert width == None


def test_INBD_model():
    boundary    = INBD.Boundary.from_polar_coordinates(np.ones(50), np.linspace(-np.pi, np.pi, 50), [0,0])
    segoutput = segmentation.SegmentationOutput( *[np.ones([100,100])]*4 )
    

    for kw in [
        {'wedging_rings':True},
        {'wedging_rings':False},
        {'wedging_rings':False, 'concat_radii':True, 'backbone':'resnet18'},
        {'wedging_rings':False, 'concat_radii':True, 'backbone':'mobilenet3l'},
    ]:
        class Mock:
            scale = 4
        model  = INBD.INBD_Model(segmentationmodel=Mock, **kw)
        pgrid  = INBD.PolarGrid.construct(np.ones([3,100,100]), segoutput, None, boundary, width=50, concat_radii=kw.get('concat_radii', False))
        y      = model.forward_from_polar_grid(pgrid)
        assert y['x'].shape[-2:]    == (256,50)
        
        if kw['wedging_rings']:
            assert y['wd_x'].shape[-2:] == (1,50)

        new_boundary = model.output_to_boundary(y['x'][0], boundary, pgrid)

def test_INBD_save_model():
    tmpdir        = tempfile.TemporaryDirectory(prefix='delete_me_')
    segmodel      = segmentation.SegmentationModel(backbone='resnet18', downsample_factor=4)
    segmodelpath  = os.path.join(tmpdir.name, 'segmodel.pt.zip')
    segmodel.save(segmodelpath)
    #reload segmentation model
    segmodel      = util.load_segmentationmodel(segmodelpath)
    model         = INBD.INBD_Model(segmentationmodel=segmodel)

    filename0 = model.save(os.path.join(tmpdir.name, '%Y-%m-%d_model'))
    assert os.path.exists(filename0)
    assert filename0.endswith('.pt.zip')
    assert '%Y' not in filename0

    model_reloaded = util.load_model(filename0)
    assert model_reloaded
    assert 'INBD_Model' in str(model_reloaded.__class__)
    assert model_reloaded.scale == model.scale
    assert model.segmentationmodel[0] is not None

    #save again
    filename1 = model_reloaded.save(os.path.join(tmpdir.name, '%Y-%m-%d_model2'))
    assert os.path.exists(filename1)

    tmpdir.cleanup()




def test_select_largest_connected_component():
    x = np.zeros([100,100])
    x[20:50, 20:50] = 1  #largest component
    x[25, 25]       = 0  #with a hole
    x[60:65, 60:65] = 1
    x[75:80, 75:80] = 1

    y = util.select_largest_connected_component( x>0 )
    assert np.all(y[20:50,20:50] == 1) #hole must be filled
    
    import scipy.ndimage
    assert scipy.ndimage.label(y)[1] == 1

    x = np.zeros([100,100])
    y = util.select_largest_connected_component( x>0 )
    #assert no error

def test_clip_boundary():
    boundary0 = INBD.Boundary.from_polar_coordinates(np.ones(50)*10, np.linspace(-np.pi, np.pi, 50), [0,0])
    boundary1 = INBD.Boundary.from_polar_coordinates(np.ones(50)*5,  np.linspace(-np.pi, np.pi, 50), [0,0])

    boundary2 = boundary1.clip_to_previous_boundary(boundary0)
    assert np.all(boundary2.compute_radii() >= 10)


def test_mask_out_interpolate_boundary():
    radii0  = np.linspace(1,10,50)
    boundary0 = INBD.Boundary.from_polar_coordinates(radii0, np.linspace(-np.pi, np.pi, 50), [0,0])

    mask    = np.ones(50, dtype=bool)
    mask[-10:] = 0


    boundary1 = boundary0.mask_out_interpolate_boundary(mask, circular=False)
    radii1  = boundary1.compute_radii()

    assert np.allclose( np.diff(radii1[-10:]), 0 )
    assert radii1[-1] == radii0[-11]
    assert np.allclose( boundary1.compute_angles(), boundary0.compute_angles() )

    boundary2 = boundary0.mask_out_interpolate_boundary(mask, circular=True)
    radii2  = boundary2.compute_radii()

    assert radii2[-1] < radii0[-11]
    assert np.all( np.diff(radii2[-10:]) < 0 )
    assert len(radii2) == len(radii0)
    assert np.allclose( boundary2.compute_angles(), boundary0.compute_angles() )

    mask1    = np.ones(50, dtype=bool)
    mask1[:10] = 0
    boundary3 = boundary0.mask_out_interpolate_boundary(mask1, circular=True)
    radii3  = boundary3.compute_radii()

    assert radii3[0] > radii0[10]
    assert np.allclose( boundary3.compute_angles(), boundary0.compute_angles() )


def test_wedging_ring_target():
    l  = np.random.randint(1,10)
    i0 = np.random.randint(80)
    i1 = i0 + np.random.randint(2,20)
    class polargrid_mock:
        annotation = np.ones([1,256,100]) * (l+1)
        annotation[..., i0:i1] = l+2
    tgt = INBD.wedging_ring_target(polargrid_mock, l)
    assert tgt.shape == (100,)
    assert np.all(tgt[i1:] == 1)
    assert np.all(tgt[:i0] == 1)
    assert np.all(tgt[i0:i1] == 0)

def test_boundaries_to_svg():
    boundaries = [
        INBD.Boundary.from_polar_coordinates(np.ones(50)*5,  np.linspace(-np.pi, np.pi, 50), [50,50]),
        INBD.Boundary.from_polar_coordinates(np.ones(50)*10, np.linspace(-np.pi, np.pi, 50), [50,50]),
        INBD.Boundary.from_polar_coordinates(np.ones(50)*20, np.linspace(-np.pi, np.pi, 50), [50,50]),
    ]

    svg = INBD.boundaries_to_svg(boundaries, (200,150))
    #make sure it's valid xml
    from xml.dom.minidom import parseString
    dom = parseString(svg)

    #no detected boundaries, make sure there is no error
    svg = INBD.boundaries_to_svg([], (200,150))
    #make sure it's valid xml
    from xml.dom.minidom import parseString
    dom = parseString(svg)
    
    