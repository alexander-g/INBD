from src import datasets
import numpy as np
import PIL.Image
import tempfile, os


def test_slice_stitch_images():
    x         = np.random.random([3,1024,2048])
    slack     = np.random.randint(20,50)
    patchsize = np.random.randint(100,400)
    patches   = datasets.slice_into_patches_with_overlap(x, patchsize, slack)
    
    assert patches[0].shape == (3, patchsize, patchsize)

    y         = datasets.stitch_overlapping_patches(patches, x.shape, slack=slack)

    assert x.shape == y.shape
    assert np.all( x == y.numpy() )


def test_load_annotation_rgb():
    x         = np.zeros([1000,1000,4], dtype='uint8')
    x[...,-1] = 255
    c         = np.random.randint(300,700, size=2)
    x[c[0]-200:, c[1]-200:][:200*2, :200*2] = (100,100,200,255)
    x[c[0]-101:, c[1]-101:][:101*2, :101*2] = (255,255,255,255) #boundary
    x[c[0]-100:, c[1]-100:][:100*2, :100*2] = (255,  0,  0,255) #red
    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me')
    imgf   = os.path.join(tmpdir.name, 'img0.png')
    PIL.Image.fromarray(x).save(imgf)

    annotation = datasets.load_instanced_annotation(imgf, force_cpu=True)
    assert np.any(annotation==0)
    assert annotation[0,0]        == -1
    assert annotation[c[0], c[1]] == 1
    assert annotation.max()       >  1


def test_load_annotation():
    x         = np.zeros([1000,1000], dtype='int32')
    x[:]      = -1
    c         = np.random.randint(300,700, size=2)
    x[c[0]-200:, c[1]-200:][:200*2, :200*2] = 2
    x[c[0]-101:, c[1]-101:][:101*2, :101*2] = 0 #boundary
    x[c[0]-100:, c[1]-100:][:100*2, :100*2] = 1 #center
    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me')
    imgf   = os.path.join(tmpdir.name, 'img0.tiff')
    PIL.Image.fromarray(x, 'I').save(imgf)

    annotation = datasets.load_instanced_annotation(imgf)
    assert np.any(annotation==0)
    assert annotation[0,0]        == -1
    assert annotation[c[0], c[1]] == 1
    assert annotation.max()       >  1

    annotation = datasets.load_instanced_annotation(imgf, downscale=2.0)
    assert annotation.shape == (500,500)


