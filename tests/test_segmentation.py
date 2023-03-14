from src import segmentation, util
import tempfile, os
import numpy as np
import PIL.Image
import torch, torchvision



def test_segmentationdataset():
    x        = np.zeros([100,100], dtype='int32')
    x[:10]   = -1             #background
    x[10]    = 0              #boundary
    x[11:20] = 2              #2nd ring
    x[20]    = 0              #boundary
    x[21:30] = 1              #first ring
    x[30]    = 0              #red:   boundary
    x[31:40] = 2              #2nd ring
    x[40]    = 0              #boundary
    x[41:]   = -1             #background

    tmpdir = tempfile.TemporaryDirectory()
    tgtf    = os.path.join(tmpdir.name, 'annotation.tiff')
    PIL.Image.fromarray(x, 'I').save(tgtf)

    y_true  = segmentation.SegmentationDataset.load_targetfile(tgtf, one_hot=False).numpy()
    assert y_true.shape == (1,100,100)
    assert np.all(y_true[0,:10]  == segmentation.CLASSES['background'])
    assert np.all(y_true[0,41:]  == segmentation.CLASSES['background'])
    assert np.all(y_true[0,10]   == segmentation.CLASSES['boundary'])
    assert np.all(y_true[0,20]   == segmentation.CLASSES['boundary'])
    assert np.all(y_true[0,30]   == segmentation.CLASSES['boundary'])
    assert np.all(y_true[0,40]   == segmentation.CLASSES['boundary'])
    assert np.all(y_true[0,21:30] == segmentation.CLASSES['center'])

    y_true  = segmentation.SegmentationDataset.load_targetfile(tgtf, one_hot=True).numpy()
    assert y_true.shape == (1, 100,100)



def test_segmentationtask():
    x          = torch.ones([2,3,100,100])
    ytrue      = torch.ones([2,len(segmentation.CLASSES),100,100])
    
    class MockModule(torch.nn.Sequential):
        def forward(self, x, sigmoid):
            y = super().forward(x)
            if sigmoid:
                y = torch.sigmoid( y )
            return dict( [ (cls, y[:,i]) for cls,i in segmentation.CLASSES.items()] )
    mockmodule = MockModule(torch.nn.Conv2d(3,4,1))

    task       = segmentation.SegmentationTask(mockmodule)
    loss, logs = task.training_step(batch=(x,ytrue))

def test_segmentationmodel():
    m      = segmentation.SegmentationModel(downsample_factor=2)
    x      = np.random.randint(255, size=[100,100,3], dtype='uint8')
    tmpdir = tempfile.TemporaryDirectory()
    imgf   = os.path.join(tmpdir.name, 'img.jpg')
    img    = PIL.Image.fromarray(x)
    img.save(imgf)
    imgT   = torchvision.transforms.ToTensor()(img)

    m.process_image(x)     #numpy
    m.process_image(imgf)  #filename
    m.process_image(img)   #PIL
    m.process_image(imgT)  #tensor

    y = m.process_image(x, upscale_result=True)
    assert y[0].shape == (100,100)
    
    y = m.process_image(x, upscale_result=False)
    assert y[0].shape == ( 50, 50)

    y = m.process_image(x, upscale_result=True, downscale=1)
    assert y[0].shape == (100,100)

    y = m.process_image(x, upscale_result=False, downscale=1.2)
    assert y[0].shape == (100//1.2,100//1.2)


def test_save_model():
    model      = segmentation.SegmentationModel(backbone='resnet18', downsample_factor=4)

    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me_')
    filename0 = model.save(os.path.join(tmpdir.name, '%Y-%m-%d_model'))
    assert os.path.exists(filename0)
    assert filename0.endswith('.pt.zip')
    assert '%Y' not in filename0

    import zipfile
    with zipfile.ZipFile(filename0) as zipf:
        files = zipf.namelist()
        assert len( [f for f in files if 'segmentation.py' in f] ) > 0

    model_reloaded = util.load_model(filename0)
    assert model_reloaded
    assert 'SegmentationModel' in str(model_reloaded.__class__)
    assert model_reloaded.scale == model.scale

    #save again
    filename1 = model_reloaded.save(os.path.join(tmpdir.name, '%Y-%m-%d_model2'))
    assert os.path.exists(filename1)

    tmpdir.cleanup()
