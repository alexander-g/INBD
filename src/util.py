import time, os, glob, shutil, sys
import typing as tp
import numpy as np
import scipy
import torch

def select_largest_connected_component(x:np.ndarray) -> np.ndarray:
    '''Remove all connected components from binary array mask except the largest'''
    x_labeled      = scipy.ndimage.label(x)[0]
    labels,counts  = np.unique(x_labeled[x_labeled!=0], return_counts=True)
    if len(labels) == 0:
        return x
    maxlabel       = labels[np.argmax(counts)]
    return scipy.ndimage.binary_fill_holes( x_labeled == maxlabel )

def filter_labelmap(labelmap:np.ndarray, threshold=0.001) -> np.ndarray:
    N              = np.prod( labelmap.shape )
    labels, counts = np.unique( labelmap, return_counts=True )
    result         = labelmap.copy()
    for l,c in zip(labels, counts):
        if c/N < threshold:
            result[labelmap==l] = 0
    return result


def backup_code(destination:str) -> str:
    destination = time.strftime(destination)
    cwd      = os.path.realpath(os.getcwd())+'/'
    srcfiles = glob.glob('src/**/*.py', recursive=True) + ['main.py']
    for src_f in srcfiles:
        src_f = os.path.realpath(src_f)
        dst_f = os.path.join(destination, src_f.replace(cwd, ''))
        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
        shutil.copy(src_f, dst_f)
    open(os.path.join(destination, 'args.txt'), 'w').write(' '.join(sys.argv))
    return destination

def output_name(args):
    reso = f'a{args.angular_density:.1f}' if args.modeltype=='INBD' else f'x{args.downsample}'
    name = f'%Y-%m-%d_%Hh%Mm%Ss_{args.modeltype}_{args.epochs}e_{reso}_{args.suffix}'
    name = os.path.join(args.output, name)
    name = time.strftime(name)
    return name

def load_model(path:str):
    importer = torch.package.PackageImporter(path)
    model    = importer.load_pickle('model', 'model.pkl')
    return model


def _infer_segmentationmodel_backbone_name(model):
    backbone = getattr(model, 'backbone_name', None)
    if backbone is None:
        if 'Hardswish' in str(model):
            backbone = 'mobilenet3l'
        else:
            backbone = 'resnet18'
    return backbone

def load_segmentationmodel(path):
    from . import segmentation
    importer = torch.package.PackageImporter(path)
    model    = importer.load_pickle('model', 'model.pkl')
    backbone = _infer_segmentationmodel_backbone_name(model)
    state    = model.state_dict()
    model    = segmentation.SegmentationModel(downsample_factor=model.scale, backbone=backbone)
    model.load_state_dict(state)
    return model


def read_splitfile(splitfile:str) -> tp.List[str]:
    files       = open(splitfile).read().strip().split('\n')
    if files == ['']:
        return []
    
    dirname     = os.path.dirname(splitfile)
    files       = [f if os.path.isabs(f) else os.path.join(dirname, f) for f in files]
    assert all([os.path.exists(f) for f in files])
    return files


def read_splitfiles(images_splitfile:str, annotations_splitfile:str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    imagefiles  = read_splitfile(images_splitfile)
    annotations = read_splitfile(annotations_splitfile)
    assert len(imagefiles) == len(annotations), [len(imagefiles), len(annotations)]
    return imagefiles, annotations


def labelmap_to_areas_output(labelmap:np.ndarray) -> str:
    output        = ''
    labels,counts = np.unique(labelmap[labelmap>0], return_counts=True)
    for l,c in zip(labels, counts):
        output += f'{l}, {c}\n'
    return output