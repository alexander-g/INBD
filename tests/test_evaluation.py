from src import evaluation
import numpy as np
import torch

def test_iou():
    a = np.zeros(10, bool)
    a[:5] = 1
    b = np.zeros(10, bool)
    b[3:] = 1

    assert evaluation.IoU(a,b) == 0.2

def test_miou():
    a = torch.zeros([2,10]).bool()
    a[0,:5] = 1
    b = torch.zeros([2,10]).bool()
    b[0,3:] = 1

    assert evaluation.IoU(a,b) == 0.2


def test_evaluate_single_result():
    labelmap_result     = np.zeros([100,100])
    labelmap_result[20:80, 20:30] = 4
    labelmap_result[30:70, 30:70] = 2
    labelmap_result[40:60, 40:60] = 1
    
    labelmap_annotation = np.zeros([100,100])
    labelmap_annotation[20:80, 20:80] = 3
    labelmap_annotation[30:70, 30:70] = 2
    labelmap_annotation[40:60, 40:60] = 1

    metrics = evaluation.evaluate_single_result(labelmap_result, labelmap_annotation, iou_threshold=0.5)
    assert metrics['TP'] == 2
    assert metrics['FP'] == 1
    assert metrics['FN'] == 1
    #assert metrics['mIoU'] == 1.0


    metrics = evaluation.evaluate_single_result(labelmap_result, labelmap_annotation, iou_threshold=0.2)
    assert metrics['TP'] == 3
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0
    #assert metrics['mIoU'] < 1.0


def test_combine_metrics_at_iou_levels():
    mock_metrics = [
        {0.50: {'TP':7, 'FP':5, 'FN':3, 'recall':0.7 }, 0.75:{'TP':7, 'FP':5, 'FN':3, 'recall': 0.4}, 'ARAND':0.0 },
        {0.50: {'TP':7, 'FP':5, 'FN':3, 'recall':0.2 }, 0.75:{'TP':7, 'FP':5, 'FN':3, 'recall': 0.0}, 'ARAND':0.0 },
    ]

    combined = evaluation.combine_metrics_at_iou_levels(mock_metrics, iou_levels=[0.5, 0.75])
    assert np.allclose( combined['mAR'], (1.1/2 + 0.2/2)/2 )

