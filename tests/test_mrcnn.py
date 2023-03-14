import tempfile, os
from src import maskrcnn, util
import numpy as np
import torch



def test_labelmap_to_boxes():
    x = np.zeros([500,500], int)
    x[100:-100, 110:-100] = 5
    x[200:-200, 200:-200] = 4
    x[220:-220, 220:-220] = 1

    y = maskrcnn.labelmap_to_boxes(x)

    assert len(y['boxes']) == 3
    assert len(y['masks']) == 3
    assert y['boxes'][-1].numpy().tolist() == [110, 100, 400,400]
    
    assert np.all(y['masks'][1].numpy() == (x==4))


def test_mrcnn_output_to_labelmap():
    m0                      = torch.zeros([1,500,500])
    m0[:, 200:300, 200:300] = 1
    m1                      = torch.zeros([1,500,500])
    m1[:, 100:400, 100:400] = 1

    x   = {
        'boxes' : torch.as_tensor([
            [200,200,  300,300],
            [100,100,  400,400],
        ]),
        'masks' : torch.stack([m0, m1]),
        'scores': [1,1],
        'labels': [1,1],
    }

    labelmap = maskrcnn.mrcnn_output_to_labelmap(x, (500,500))
    assert labelmap[200,200] == 2
    assert labelmap[100,100] == 1

