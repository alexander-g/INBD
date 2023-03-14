import os, sys, time
import typing as tp
import numpy as np
import PIL.Image
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter

from . import datasets, training

MODULES = []

class UNet(torch.nn.Module):
    '''Backboned U-Net'''

    class UpBlock(torch.nn.Module):
        def __init__(self, in_c, out_c, inter_c=None):
            #super().__init__()
            torch.nn.Module.__init__(self)
            inter_c        = inter_c or out_c
            self.conv1x1   = torch.nn.Conv2d(in_c, inter_c, 1)
            self.convblock = torch.nn.Sequential(
                torch.nn.Conv2d(inter_c, out_c, 3, padding=1, bias=0),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(),
            )
        def forward(self, x:torch.Tensor, skip_x:torch.Tensor, relu=True) -> torch.Tensor:
            x = torch.nn.functional.interpolate(x, skip_x.shape[2:])   #TODO? mode='bilinear
            x = torch.cat([x, skip_x], dim=1)
            x = self.conv1x1(x)
            x = self.convblock(x)
            return x
    
    def __init__(self, backbone='mobilenet3l', out_channels=1, downsample_factor=1, backbone_pretrained:bool=True):
        torch.nn.Module.__init__(self)
        factory_func = BACKBONES.get(backbone, None)
        if factory_func is None:
            raise NotImplementedError(backbone)
        self.backbone, C = factory_func(backbone_pretrained)
        self.backbone_name = backbone
        self.scale       = downsample_factor
        
        self.up0 = self.UpBlock(C[-1]    + C[-2],  C[-2])
        self.up1 = self.UpBlock(C[-2]    + C[-3],  C[-3])
        self.up2 = self.UpBlock(C[-3]    + C[-4],  C[-4])
        self.up3 = self.UpBlock(C[-4]    + C[-5],  C[-5])
        self.up4 = self.UpBlock(C[-5]    + 3,      32)
        self.cls = torch.nn.Conv2d(32, out_channels, 3, padding=1)
    
    def forward(self, x:torch.Tensor, sigmoid=False, return_features=False) -> torch.Tensor:
        device = list(self.parameters())[0].device
        x      = x.to(device)
        
        X = self.backbone(x)
        X = ([x] + [X[f'out{i}'] for i in range(5)])[::-1]
        x = X.pop(0)
        x = self.up0(x, X[0])
        x = self.up1(x, X[1])
        x = self.up2(x, X[2])
        x = self.up3(x, X[3])
        x = self.up4(x, X[4])
        if return_features:
            return x
        x = self.cls(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
    
    def load_image(self, path:str) -> np.ndarray:
        return PIL.Image.open(path).convert('RGB') / np.float32(255)
    
    def process_image(
        self, 
        image:tp.Union[str, np.ndarray], 
        progress_callback                 = lambda *x:None, 
        downscale:tp.Optional[float]      = None,                #overrides self.scale if not None
        upscale_result                    = True,
        batchsize:int                     = 4, 
        **forward_kwargs,
    ) -> np.ndarray:
        if isinstance(image, str):
            image = self.load_image(image)
        x = image
        if not torch.is_tensor(x):
            x = torchvision.transforms.ToTensor()(image)
        
        imgshape  = x.shape[-2:]
        scale     = 1/(downscale or self.scale)
        x         = datasets.resize_tensor(x, scale=scale, mode='bilinear')
        patches   = datasets.slice_into_patches_with_overlap(x)
        with torch.no_grad():
            output_patches = []
            for i in range(0, len(patches), batchsize):
                progress_callback( i / len(patches) )
                batch   = torch.stack(patches[i:][:batchsize])
                output_patches += list(self.eval().forward(batch, **forward_kwargs))
        output  = datasets.stitch_overlapping_patches(output_patches, x.shape)
        if upscale_result:
            output  = datasets.resize_tensor(output, size=imgshape, mode='bilinear')
        result  = output.cpu().numpy()

        progress_callback(1.0)
        return result
    
    def save(self, destination ):
        if isinstance(destination, str):
            destination = time.strftime(destination)
            if not destination.endswith('.pt.zip'):
                destination += '.pt.zip'
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            import torch_package_importer as imp
            #re-export
            importer = (imp, torch.package.sys_importer)
        except ImportError as e:
            #first export
            importer = (torch.package.sys_importer,)
        with torch.package.PackageExporter(destination, importer) as pe:
            #save all python files in src folder
            interns = [k for k in sys.modules.keys() if 'src.' in k and not 'cython.' in k]
            pe.intern(interns)
            pe.extern('**', exclude=['torchvision.**'])
            externs = ['torchvision.ops.**', 'torchvision.datasets.**', 'torchvision.io.**']
            pe.intern('torchvision.**', exclude=externs)
            pe.extern(externs)
            
            pe.save_pickle('model', 'model.pkl', self.cpu())
        return destination

    def start_training(self, 
                       imagefiles_train,      targetfiles_train,
                       imagefiles_valid=None, targetfiles_valid=None,
                       epochs      = 'auto', 
                       callback    = None, 
                       num_workers = 'auto',
                       batch_size  = 8,
                       ds_cls      = datasets.Dataset,
                       task_cls    = training.TrainingTask,
                       scales      = (1.8, 3.0),
                       ds_kwargs   = {},
                       **task_kwargs
        ):
        task     = task_cls(self, epochs=epochs, callback=callback, **task_kwargs)
        if isinstance(imagefiles_train, datasets.Dataset) or 'Dataset' in str(type(imagefiles_train)):
            ds_train = imagefiles_train
        else:
            ds_train = ds_cls(imagefiles_train, targetfiles_train, 
                              scale_range=scales,    augment=True, 
                              **ds_kwargs)
        ld_train = ds_train.create_dataloader(batch_size, shuffle=True, num_workers=num_workers)
        
        ld_valid = None
        if imagefiles_valid is not None and targetfiles_valid is not None:
            ds_valid = ds_cls(imagefiles_valid, targetfiles_valid,
                              scale_range=[self.scale], augment=False, 
                              **ds_kwargs)
            ld_valid = ds_valid.create_dataloader(batch_size, shuffle=False, num_workers=num_workers)
        
        self.requires_grad_(True)
        rc = task.fit(ld_train, ld_valid, epochs=epochs)
        self.eval().cpu().requires_grad_(False)
        return rc
    


def resnet18_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet18(pretrained=pretrained)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 64, 128, 256, 512]
    return backbone, channels

def resnet50_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet50(pretrained=pretrained)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 256, 512, 1024, 2048]
    return backbone, channels

def mobilenet3l_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
    return_layers = {'1':'out0', '3':'out1', '6':'out2', '10':'out3', '16':'out4'}
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [16, 24, 40, 80, 960]
    return backbone, channels

BACKBONES = {
    'resnet18':    resnet18_backbone,
    'resnet50':    resnet50_backbone,
    'mobilenet3l': mobilenet3l_backbone,
}
