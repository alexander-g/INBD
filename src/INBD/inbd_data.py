import typing as tp, itertools
import numpy as np
import scipy
import torch, torchvision
import skimage
import networkx as nx
import PIL.Image

from ..             import datasets, util
from ..segmentation import SegmentationModel, SegmentationOutput


#data loading


class TrainstepData(tp.NamedTuple):
    inputimage:   torch.Tensor
    segmentation: SegmentationOutput
    annotation:   np.ndarray


class INBD_Dataset:
    trainstepdata:   tp.List[TrainstepData]
    img_ring_combos: tp.List[tp.Tuple[int,int]]

    def __init__(self, images:tp.List[str], annotations:tp.List[str], scale_range=(1.8, 3.0), augment=False, cachedir='./cache/'):
        super().__init__()
        self.imagefiles  = images
        #self.annotations = annotations
        self.annotations = [load_annotation_for_inbd(annf, downscale=min(scale_range)) for annf in annotations]
        self.augment     = augment
        self.scales     = scale_range
        #no caching
        #self.cachedir   = cachedir
        #self.load_and_cache_dataset(images, annotations)

    #called manually
    def load_and_cache_dataset(self, segmentationmodel:SegmentationModel):
        self.trainstepdata = []
        for imgf, ann in zip(self.imagefiles, self.annotations):
            #load image with random scale
            scale  = segmentationmodel.scale if not self.augment else np.random.uniform(*self.scales)
            image  = datasets.load_image(imgf, downscale=scale, mode='bilinear')
            if self.augment and np.random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                ann   = np.flip(ann, axis=0).copy()
            
            #process image with segmentationmodel
            output = segmentationmodel.process_image(image, downscale=1, upscale_result=False)
            if self.augment:
                image = datasets.augment_color_jitter(image)
            #scale ann to same size as input
            ann    = torch.as_tensor(ann)[None].float()
            ann    = torch.nn.functional.interpolate(ann[None], size=image.shape[-2:], mode='nearest')[0].long().numpy()
            self.trainstepdata += [ TrainstepData(image, output, ann) ]

        self.img_ring_combos = np.concatenate([
            list(zip(np.ones(1000, int)*i, np.unique( inp.annotation )[1:-1])) for i,inp in enumerate(self.trainstepdata)
        ])
    
    def __len__(self):
        return len(self.img_ring_combos)

    def __getitem__(self, i:int) -> tp.Tuple[TrainstepData, int]:
        img_i, ring_i = self.img_ring_combos[i]
        return (self.trainstepdata[img_i], ring_i)

    def __iter__(self):
        ixs = np.arange(len(self)) if not self.augment else np.random.permutation(len(self))
        for i in ixs:
            yield self[i]

    def create_dataloader(self, *a, **kw) -> 'INBD_Dataset':
        #returning self for re-load-and-caching() in the training task
        return self


def load_annotation_for_inbd(annotationfile:str, *a, **kw) -> np.ndarray:
    if annotationfile.endswith('.tiff'):
        return load_annotation_for_inbd_tiff(annotationfile, *a, **kw)
    elif annotationfile.endswith('.png'):
        return load_annotation_for_inbd_png(annotationfile, *a, **kw)
    else:
        raise NotImplementedError(annotationfile)

def load_annotation_for_inbd_tiff(annotationfile:str, *a, **kw) -> np.ndarray:
    a = datasets.load_instanced_annotation(annotationfile, *a, **kw)
    a = remove_boundary_class(a)
    return a

def load_annotation_for_inbd_png(annotationfile:str, *a, **kw) -> np.ndarray:
    L      = datasets.load_instanced_annotation(annotationfile, *a,  white_label=0, black_label=-1, red_label=1, **kw)
    annotation_sanity_check(L, annotationfile)
    L      = remove_boundary_class(L)
    chain  = labelmap_to_chain(L, start_class=1, bg_class=-1)
    L      = relabel(L, chain)
    return L



def remove_boundary_class(labelmap:np.ndarray, boundaryclass:int=0, bg_class:int=-1) -> np.ndarray:
    '''Remove the class boundaryclass from a labeled array, (so that the tree ring instances touch each other)'''
    boundarymask   = (labelmap==boundaryclass)
    backgroundmask = (labelmap==bg_class)
    result                 = labelmap.copy()
    result[boundarymask]   = 0
    result[backgroundmask] = 0
    while np.any( result[boundarymask]==0 ):
        result = skimage.segmentation.expand_labels(result, distance=100)
    result[backgroundmask] = bg_class
    return result

def labelmap_to_chain(labelmap:np.ndarray, start_class:int, bg_class=-1) -> nx.Graph:
    labelmap = np.where(labelmap==bg_class, labelmap.max()+1, labelmap)
    bg_class = labelmap.max()
    rag      = skimage.future.graph.rag_boundary(labelmap, np.zeros(labelmap.shape))
    if bg_class in rag.nodes:
        rag.remove_node(bg_class)
    return rag_to_chain(rag, labelmap, start_class)

def find_label_boundary(labelmap:np.ndarray, l0:int, l1:int) -> np.ndarray:
    '''Get the boundary between two touching tree rings l0 & l1'''
    mask     = np.isin(labelmap, [l0,l1])
    boundary = skimage.segmentation.find_boundaries(labelmap * mask)
    boundary = boundary * skimage.morphology.erosion(mask)
    return boundary

def rag_to_chain(rag:nx.Graph, labelmap:np.ndarray, start_class:int) -> nx.Graph:
    '''Convert a RAG to a chain, also considering wedging rings'''
    path   = [start_class]
    center = np.argwhere(labelmap == start_class).mean(0)
    
    while 1:
        l             = path[-1]
        neighbors     = set( rag.neighbors(l) ).difference(path)
        if   len(neighbors) == 0:
            break
        elif len(neighbors) == 1:
            path.append(list(neighbors)[0])
        else:
            #wedging ring
            rejected = []
            for l0,l1 in itertools.combinations(neighbors, 2):
                boundary = find_label_boundary(labelmap, l0,l1)
                boundary = np.argwhere(boundary)
                points = np.linspace(boundary, center, 100).reshape(-1,2)
                inbetween         = scipy.ndimage.map_coordinates(labelmap, points.T, order=0)
                inbetween_counts  = dict( zip(*np.unique( inbetween, return_counts=True )) )
                
                l0_counts = inbetween_counts.get(l0, 0)
                l1_counts = inbetween_counts.get(l1, 0)
                
                if l0_counts == l1_counts == 0:
                    continue
                
                rejected.append( l0 if l0_counts < l1_counts else l1 )
            neighbors = neighbors.difference(rejected)
            assert len(neighbors) == 1
            path.append(list(neighbors)[0])
    chain = rag.copy()
    chain.remove_edges_from( list(rag.edges) )
    chain.add_path(path)
    return chain

def relabel(labelmap:np.ndarray, chain:nx.Graph) -> np.ndarray:
    '''Sort labels in the labelmap according to chain'''
    endpoints = [n for n in chain.nodes if len(chain.adj[n])==1]
    assert len(endpoints) == 2
    origin    = endpoints[0]
    end       = endpoints[-1]
    
    order     =  nx.shortest_path(chain, origin, end)
    result    = labelmap.copy()
    for i,l in enumerate(order, 1):
        result[labelmap==l] = i
    return result


