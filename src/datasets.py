import glob, os, tempfile
import typing as tp
import numpy as np
import scipy.ndimage
import PIL.Image
import torch, torchvision


class Dataset:
    def __init__(
        self, 
        images, 
        targets, 
        scale_range             =   (1.8, 3.0), 
        patch_size              =   512, 
        slack_size              =   32, 
        augment                 =   False, 
        color_jitter            =   True,
        cachedir                =   './cache/'
    ):
        super().__init__()
        self.scales     = scale_range
        self.patch_size = patch_size
        self.slack_size = slack_size
        self.cachedir   = cachedir
        self.load_and_cache_dataset(images, targets)
        self.augment      = augment
        self.color_jitter = color_jitter
    
    def __len__(self):
        return len(self.images)
    
    def _create_cache_dir(self):
        os.makedirs(self.cachedir, exist_ok=True)
        self.cachedir   = tempfile.TemporaryDirectory(dir=self.cachedir)
        print('Caching files into temporary directory:', self.cachedir.name)
    
    def load_and_cache_dataset(self, imagefiles:tp.List[str], targetfiles:tp.List[str]) -> None:
        '''Load data, split into patches and cache in a temporary directory for faster training'''
        self._create_cache_dir()
        for imagefile, targetfile in zip(imagefiles, targetfiles):
            img_full = PIL.Image.open(imagefile).convert('RGB')
            img_full = np.asarray(img_full).transpose(2,0,1)  #CHW ordering, uint8
            tgt_full = self.load_targetfile(targetfile)
            assert len(tgt_full.shape) == 3

            patch_size = int(self.patch_size  * max(self.scales))
            slack      =          patch_size // 2
            base       = os.path.basename(imagefile)
            for i,patch in enumerate(slice_into_patches_with_overlap(img_full, patch_size, slack)):
                np.save(os.path.join(self.cachedir.name, f'img_{base}_{i}.npy'), patch)
            for i,patch in enumerate(slice_into_patches_with_overlap(tgt_full, patch_size, slack)):
                    np.save(os.path.join(self.cachedir.name, f'tgt_{base}_{i}.npy'), patch)

        self.images  = sorted(glob.glob(os.path.join(self.cachedir.name, f'img_*.npy')))
        self.targets = sorted(glob.glob(os.path.join(self.cachedir.name, f'tgt_*.npy')))

    
    @staticmethod
    def load_targetfile(targetfile:str) -> torch.Tensor:
        x = PIL.Image.open(targetfile).convert('RGBA')
        x = (np.array(x) == np.uint(255)).all(-1)
        x = torchvision.transforms.ToTensor()(x).float()
        return x
    
    def _load_cached(self, i:int):
        img = np.load(self.images[i])
        tgt = np.load(self.targets[i])
        return img, tgt

    def __getitem__(self, i:int):
        img, tgt = self._load_cached(i)

        if self.augment:
            box = generate_random_box(img.shape[-2:], (self.patch_size*self.scales[0], self.patch_size*self.scales[1]), skew=(0.7, 1.2))
            box = box.astype(int)
            img = img[:, box[0]:box[2], box[1]:box[3]]
            tgt = tgt[:, box[0]:box[2], box[1]:box[3]]
        
        img = torch.as_tensor(img).float() / 255
        img = resize_tensor(img, size=self.patch_size, mode='bilinear')
        tgt = torch.as_tensor(tgt)
        tgt = resize_tensor(tgt.float(), size=self.patch_size, mode='area')
        tgt = (tgt > 0.0).float()
        
        if self.augment and np.random.random() < 0.5:
            img, tgt = torch.flip(img, dims=(-1,)), torch.flip(tgt, dims=(-1,))
        if self.augment:
            k     = np.random.randint(4)
            img   = torch.rot90(img, k, (-2,-1))
            tgt   = torch.rot90(tgt, k, (-2,-1))
        if self.augment and self.color_jitter:
            img   = augment_color_jitter(img)
        return img, tgt

    def create_dataloader(self, batch_size, shuffle=False, num_workers='auto'):
        if num_workers == 'auto':
            num_workers = os.cpu_count()
        return torch.utils.data.DataLoader(self, batch_size, shuffle, collate_fn=getattr(self, 'collate_fn', None),
                                           num_workers=num_workers, pin_memory=True,
                                           worker_init_fn=lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x) )


def generate_random_box(container_shape:tuple, target_size=(512*1.8, 512*3.0), skew=(0.7, 1.2) ) -> np.ndarray:
    container_shape = np.array(container_shape)
    shape           = np.random.uniform(*target_size) * np.random.uniform(*skew, size=2)
    shape           = np.minimum(container_shape-1, shape)
    yx0             = np.random.randint(container_shape - shape)
    yx1             = yx0 + shape
    return np.concatenate([yx0, yx1])

def augment_color_jitter(image:torch.Tensor) -> torch.Tensor:
    return torchvision.transforms.ColorJitter(
        brightness = (0.7, 1.3),
        contrast   = (0.5, 1.4),
        saturation = (0.5, 1.4),
        hue        = (-0.05, 0.05),
    )(image)


def resize_tensor(
    x:      torch.Tensor,
    mode:   tp.Union["nearest", "bilinear", "area"],
    size:   tp.Union[int, tp.Tuple[int, int], None]     =   None,
    scale:  tp.Union[float, None]                       =   None,
) -> torch.Tensor:
    assert torch.is_tensor(x), 'resize_tensor() did not receive a torch.Tensor'
    assert len(x.shape) == 3, f'resize_tensor() received unexpected shape: {x.shape}'
    assert size is not None or scale is not None
    y = torch.nn.functional.interpolate(x[None], size, scale, mode=mode)[0]
    return y


def load_image(file:str, downscale:float=1.0, mode='bilinear', colorspace='RGB') -> tp.Union[np.ndarray, torch.Tensor]:
    image = PIL.Image.open(file).convert(colorspace)
    image = torchvision.transforms.ToTensor()(image)
    if downscale != 1.0:
        image = resize_tensor(image, scale=1/downscale, mode=mode)
    return image



def load_instanced_annotation(annotationfile:str, *a, **kw) -> np.ndarray:
    if annotationfile.endswith('.tiff'):
        a = load_instanced_annotation_tiff(annotationfile, *a, **kw)
    elif annotationfile.endswith('.png'):
        a = load_instanced_annotation_rgb(annotationfile, *a, **kw)
    else:
        raise NotImplementedError(annotationfile)
    annotation_sanity_check(a, annotationfile)
    return a

def load_instanced_annotation_rgb(
    file:str, downscale:float=1.0, white_label=0, black_label=-1, red_label=1, force_cpu=False
) -> np.ndarray:
    '''Load an annotation .png file with rings as integer labels (deprecated, use tiff)'''
    image          = PIL.Image.open(file).convert('RGBA')
    image          = image.resize( [int(image.size[0]//downscale), int(image.size[1]//downscale)], 0 ) #0:nearest
    image          = np.array( image ) #uint8
    #view as int32 for faster processing
    image          = image.view('int32')[...,0]
    if not torch.cuda.is_available() or force_cpu:
        colors, counts = np.unique(image, return_counts=True)
    else:
        #a bit faster on gpu
        colors, counts = torch.unique( torch.as_tensor(image, device='cuda'), return_counts=True )
        colors         = colors.cpu().numpy()
        counts         = counts.cpu().numpy()
    colors         = [color for color,count in zip(colors, counts) if count>10]
    BLACK          = np.array([  0,  0,  0,255], 'uint8').view('int32')
    WHITE          = np.array([255,255,255,255], 'uint8').view('int32')
    RED            = np.array([255,  0,  0,255], 'uint8').view('int32')
    labelmap       = np.zeros(image.shape[:2], 'int8') #int8 to save RAM
    for i,c in enumerate(colors, red_label+1):
        if c == BLACK:
            l = black_label
        elif c == WHITE:
            l = white_label
        elif c == RED:
            l = red_label
        else:
            l = i
        mask           = (image == c)
        labelmap[mask] = l
    return labelmap

def load_instanced_annotation_tiff(file:str, downscale:float=1.0) -> np.ndarray:
    '''Load an annotation .tiff file with rings as integer labels'''
    assert file.endswith('.tiff')
    image          = PIL.Image.open(file)
    assert image.mode == 'I'
    image          = image.resize( 
        [int(image.size[0]//downscale), int(image.size[1]//downscale)], 0 
    ) #0:nearest
    labelmap       = np.array( image ).astype('int8')
    return labelmap

def annotation_sanity_check(a, annotationfile):
    assert np.any( a == -1 ), f'Annotation file {annotationfile} contains no background'
    assert np.any( a ==  0 ), f'Annotation file {annotationfile} contains no ring boundaries'
    assert np.any( a ==  1 ), f'Annotation file {annotationfile} contains no center ring'



#Helper functions for slicing images (for CHW dimension ordering)
def grid_for_patches(imageshape:tuple, patchsize:int, slack:int) -> np.ndarray:
    H,W       = imageshape[:2]
    stepsize  = patchsize - slack
    grid      = np.stack( np.meshgrid( np.minimum( np.arange(patchsize, H+stepsize, stepsize), H ), 
                                       np.minimum( np.arange(patchsize, W+stepsize, stepsize), W ), indexing='ij' ), axis=-1 )
    grid      = np.concatenate([grid-patchsize, grid], axis=-1)
    grid      = np.maximum(0, grid)
    return grid

def slice_into_patches_with_overlap(image:torch.Tensor, patchsize=1024, slack=32) -> tp.List[torch.Tensor]:
    image     = torch.as_tensor(image)
    grid      = grid_for_patches(image.shape[-2:], patchsize, slack)
    patches   = [image[...,i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches

def stitch_overlapping_patches(patches:tp.List[torch.Tensor], imageshape:tuple, slack=32, out:torch.Tensor=None) -> torch.Tensor:
    patchsize = np.max(patches[0].shape[-2:])
    grid      = grid_for_patches(imageshape[-2:], patchsize, slack)
    halfslack = slack//2
    i0,i1     = (grid[grid.shape[0]-2,grid.shape[1]-2,(2,3)] - grid[-1,-1,(0,1)])//2
    d0 = np.stack( np.meshgrid( [0]+[ halfslack]*(grid.shape[0]-2)+[           i0]*(grid.shape[0]>1),
                                [0]+[ halfslack]*(grid.shape[1]-2)+[           i1]*(grid.shape[1]>1), indexing='ij' ), axis=-1 )
    d1 = np.stack( np.meshgrid(     [-halfslack]*(grid.shape[0]-1)+[imageshape[-2]],      
                                    [-halfslack]*(grid.shape[1]-1)+[imageshape[-1]], indexing='ij' ), axis=-1 )
    d  = np.concatenate([d0,d1], axis=-1)
    if out is None:
        out = torch.empty(patches[0].shape[:-2] + imageshape[-2:]).to(patches[0].dtype)
    for patch,gi,di in zip(patches, d.reshape(-1,4), (grid+d).reshape(-1,4)):
        out[...,di[0]:di[2], di[1]:di[3]] = patch[...,gi[0]:gi[2], gi[1]:gi[3]]
    return out
