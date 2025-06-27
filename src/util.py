import time, os, glob, shutil, sys
import typing as tp
import numpy as np
import scipy
import torch
import cv2
from shapely.geometry import Polygon, Point
import json

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


def labelmap_to_contours(inbd_labelmap:np.array, cy:int,cx:int, minimum_pixels:int =50):
    region_ids = np.unique(inbd_labelmap)
    # remove background region (id=0)
    region_ids = region_ids[region_ids > 0]
    contours_list = []

    mask = np.zeros_like(inbd_labelmap)
    region_zones = np.zeros((mask.shape[0],mask.shape[1], 3), dtype=np.uint8)
    for region in region_ids:
        region_mask = inbd_labelmap == region
        mask[region_mask] = 255
        # get contours of region

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        # Contour must have a thickness == 1
        contour = contours[0].squeeze()
        if contour.ndim == 1:
            continue
        if contour.shape[0] < minimum_pixels:
            continue

        # contour = self.make_contour_of_thickness_one(contour, inbd_labelmap)#, output_dir)
        contour_poly = Polygon(contour[:, [1, 0]].tolist())
        if not contour_poly.contains(Point(cy, cx)):
            continue

        contours_list.append(contour)

    return contours_list


def polygon_2_labelme_json(polygon_list, image_path):
    """
    Converting ch_i list object to labelme format. This format is used to store the coordinates of the rings at the image
    original resolution
    @param polygon_list: ch_i list
    @param image_path: image input path
    @return:
    - labelme_json: json in labelme format. Ring coordinates are stored here.
    """

    labelme_json = {"imagePath":str(image_path), "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None}
    for idx, polygon in enumerate(polygon_list):
        if len(polygon.shape) < 2 :
            continue

        ring = {"label":str(idx+1)}

        ring["points"] = polygon.tolist()
        ring["shape_type"] = "polygon"
        ring["flags"] = {}
        labelme_json["shapes"].append(ring)

    return labelme_json

def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)
