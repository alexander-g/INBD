import numpy as np
import scipy.optimize
import warnings, typing as tp

from . import datasets


def IoU(a:np.ndarray, b:np.ndarray) -> float:
    '''Compute the Intersection over Union of two boolean arrays'''
    return (a & b).sum() / (a | b).sum()

def mIoU(a:np.ndarray, b:np.ndarray) -> float:
    '''Mean IoU for batched inputs'''
    ious = [IoU(a_i, b_i) for a_i,b_i in zip(a,b)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean( np.asarray(ious) )

def IoU_matrix(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    a_uniques = np.unique(a[a>0])
    b_uniques = np.unique(b[b>0])

    iou_matrix = []
    for l0 in a_uniques:
        for l1 in b_uniques:
            iou = IoU( (a == l0), (b == l1) )
            iou_matrix.append(iou)
    iou_matrix = np.array(iou_matrix).reshape(len(a_uniques), len(b_uniques))
    return iou_matrix

def IoU_matrix_cuda(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    import torch
    a         = torch.as_tensor(a.astype('int32'), device='cuda')
    b         = torch.as_tensor(b.astype('int32'), device='cuda')
    a_uniques = torch.unique(a[a>0])
    b_uniques = torch.unique(b[b>0])

    iou_matrix = []
    for l0 in a_uniques:
        for l1 in b_uniques:
            iou = IoU( (a == l0), (b == l1) )
            iou_matrix.append( float(iou) )
    iou_matrix = np.asarray(iou_matrix).reshape(len(a_uniques), len(b_uniques))
    return iou_matrix


def evaluate_IoU_matrix(iou_matrix:np.ndarray, iou_threshold:float) -> tp.Dict[str, tp.Any]:
    #match highest ious with each other
    ixs0, ixs1 = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
    #check iou values
    ious       = iou_matrix[ixs0, ixs1]
    ious_ok    = (ious >= iou_threshold)
    ixs0, ixs1 = ixs0[ious_ok], ixs1[ious_ok]

    TP         = np.float32(len(ixs0))
    FP         = len(iou_matrix)    - len(ixs0)
    FN         = len(iou_matrix.T)  - len(ixs0)
    return {
        'TP'        : TP,
        'FP'        : FP,
        'FN'        : FN,
        'precision' : TP / (TP+FP),
        'recall'    : TP / (TP+FN),
    }


def evaluate_single_result(labelmap_result:np.ndarray, labelmap_annotation:np.ndarray, iou_threshold=0.5) -> tp.Dict[str, tp.Any]:
    iou_matrix = IoU_matrix(labelmap_result, labelmap_annotation)
    metrics    =  evaluate_IoU_matrix(iou_matrix, iou_threshold)
    return metrics

def evaluate_single_result_at_iou_levels(labelmap_result:np.ndarray, labelmap_annotation:np.ndarray, iou_levels=np.arange(0.5, 1.0, 0.05)) -> dict:
    iou_matrix      = IoU_matrix_cuda(labelmap_result, labelmap_annotation)
    per_iou_metrics = {}
    for th in iou_levels:
        per_iou_metrics[th]     = evaluate_IoU_matrix(iou_matrix, th)
    return per_iou_metrics


def evaluate_single_result_from_annotationfile(labelmap_result:np.ndarray, annotationfile:str, downscale:float=1.0, *a,**kw) -> dict:
    labelmap_annotation = datasets.load_instanced_annotation(annotationfile, downscale)
    return evaluate_single_result(labelmap_result, labelmap_annotation, *a, **kw)

def evaluate_set_of_files(inputfiles:tp.List[str], annotationfiles:tp.List[str], model, process_kw={}, **eval_kw) -> tp.List[dict]:
    all_metrics = []
    for imgf, tgtf in zip(inputfiles, annotationfiles):
        output  = model.process_image(imgf, **process_kw)
        metrics = evaluate_single_result_from_annotationfile(output.labelmap, tgtf, downscale=model.scale, **eval_kw)
        all_metrics.append(metrics)
    return all_metrics


def compute_ARAND(result:np.ndarray, annotation:np.ndarray) -> float:
    import skimage
    annotation = np.where(annotation < 0, 0, annotation)
    ARAND      = skimage.metrics.adapted_rand_error(annotation, result, ignore_labels=[0])[0]
    return ARAND

def evaluate_single_result_from_files_at_iou_levels(resultfile:str, annotationfile:str, iou_levels=np.arange(0.50, 1.00, 0.05) ) -> tp.Dict[tp.Any, dict]:
    import skimage
    from . import INBD
    labelmap_annotation = datasets.load_instanced_annotation(annotationfile, downscale=1)
    labelmap_annotation = INBD.remove_boundary_class(labelmap_annotation)
    labelmap_result     = np.load(resultfile)
    if labelmap_result.shape != labelmap_annotation.shape:
        labelmap_result = skimage.transform.resize(labelmap_result, labelmap_annotation.shape, order=0)
    metrics_per_iou             = evaluate_single_result_at_iou_levels(labelmap_result, labelmap_annotation, iou_levels)
    metrics_per_iou['ARAND']    = compute_ARAND(labelmap_result, labelmap_annotation)
    return metrics_per_iou

def evaluate_resultfiles(resultfiles:tp.List[str], annotationfiles:tp.List[str]):
    all_metrics = []
    for resf, annf in zip(resultfiles, annotationfiles):
        metrics = evaluate_single_result_from_files_at_iou_levels(resf, annf)
        print(resf, combine_metrics_at_iou_levels([metrics]))
        all_metrics.append(metrics)
    combined_metrics = combine_metrics_at_iou_levels(all_metrics)
    return combined_metrics, all_metrics

def combine_metrics(metrics:tp.List[dict]) -> dict:
    result = {}
    for name in ['TP', 'FP', 'FN']:
        result[name] = np.sum( [m[name] for m in metrics] )
    
    for newname, name in {'AR': 'recall'}.items():
        result[newname] = np.nanmean( [m[name] for m in metrics] )
    return result

def combine_metrics_at_iou_levels(metrics:tp.List[dict], iou_levels=np.arange(0.50, 1.00, 0.05) ) -> dict:
    all_combined    = []
    for th in iou_levels:
        this_iou_metrics  = [m[th] for m in metrics]
        this_iou_combined = combine_metrics(this_iou_metrics)
        all_combined.append(this_iou_combined)
    
    return {
        'mAR'   :   np.nanmean( [m['AR']    for m in all_combined] ),
        'ARAND' :   np.nanmean( [m['ARAND'] for m in metrics] ),
    }


