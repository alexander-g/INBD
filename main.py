import argparse, sys, os, time, glob
import warnings; warnings.simplefilter('ignore')  #pytorch is too noisy


from src import util

def train(args):
    '''Main training entry point'''
    imagefiles, annotations = util.read_splitfiles(
        args.training_images, args.training_annotations
    )
    print('Number of files for training:  ', len(imagefiles))

    val_imagefiles, val_annotations = None,None
    if args.validation_images:
        val_imagefiles, val_annotations = util.read_splitfiles(
            args.validation_images, args.validation_annotations
        )
        print('Number of files for validation:', len(val_imagefiles))

    destination = util.output_name(args)
    print('Output directory:', destination)
    util.backup_code(destination)

    if args.modeltype == 'segmentation':
        from src import segmentation
        model = segmentation.SegmentationModel(
            backbone           = args.backbone,
            downsample_factor  = args.downsample,
        )
        kw     = {'scales':[args.downsample*0.8, args.downsample*1.5]}
    elif args.modeltype == 'INBD':
        assert os.path.exists(args.segmentationmodel)
        segmentationmodel = util.load_segmentationmodel(args.segmentationmodel)
        from src import INBD
        model = INBD.INBD_Model(
            segmentationmodel, 
            backbone           = args.backbone,
            wedging_rings      = args.wd, 
            angular_density    = args.angular_density,
            concat_radii       = args.radcat,
            var_ares           = args.var_ares,
            interpolate_ambiguous = args.interpolate_ambiguous,
        )
        assert args.per_epoch_it > 0
        kw     = {
            'scales'         : [model.scale*0.9, model.scale*1.2],
            'wd_lambda'      : args.wd_lambda,
            'per_epoch_it'   : args.per_epoch_it,
            'bd_augment'     : args.bd_augment,
        }
    elif args.modeltype == 'MaskRCNN':
        from src import maskrcnn
        model = maskrcnn.MaskRCNN_RingDetector(nms = args.nms, accumulating = args.mrcnn_acc)
        kw    = {
            'scales'         : [args.downsample],
        }
    elif args.modeltype == 'cartesian':
        assert os.path.exists(args.segmentationmodel)
        segmentationmodel = util.load_segmentationmodel(args.segmentationmodel)

        from src.cartesian import cartesian
        model = cartesian.CartesianModel(segmentationmodel, input_size=args.size)
        kw    = {
            'ds_kwargs' : {
                'input_size'        : args.size,
                'segmentationmodel' : segmentationmodel,
            }
        }
    
    #save already now and immediately reload model
    #otherwise might lead to inconsistencies if code changes during training
    model_destination     = os.path.join(destination, 'model')
    model_destination_tmp = model.save(model_destination+'.tmp.pt.zip')
    model                 = util.load_model(model_destination_tmp)
    err = model.start_training(
        imagefiles,     annotations,
        val_imagefiles, val_annotations,
        epochs              = args.epochs,
        lr                  = args.lr,
        amp                 = args.amp,
        val_freq            = args.val_freq,
        **kw
    )
    if err:
        print('Aborted')
        sys.exit(1)
    
    #save again with trained weights
    model_destination = model.save( model_destination )
    os.remove(model_destination_tmp)
    print('Saved to ', model_destination)


def inference(args):
    import matplotlib.cm as mplcm, PIL.Image, numpy as np, torch

    if not os.path.exists(args.images):
        print(f'File {args.images} does not exist')
        return

    if args.images.lower().endswith('.txt'):
        imagefiles = util.read_splitfile(args.images)
    elif args.images.lower().endswith('.jpg') or args.images.lower().endswith('.jpeg'):
        imagefiles = [args.images]
    else:
        print(f'[ERROR] unknown file type: {args.images}')
        return
    print(f'Running inference on {len(imagefiles)} files')

    assert os.path.exists(args.model)
    model      = util.load_model(args.model).eval().requires_grad_(False)
    if torch.cuda.is_available():
        model.cuda()

    modelbasename = args.model.split('/')[-2]
    outputdir     = os.path.join(args.output, f'{modelbasename}_{args.suffix}' )
    os.makedirs(outputdir, exist_ok=True)
    print(f'Saving outputs to: {outputdir}')

    for i,f in enumerate(imagefiles):
        print(f'[{i:4d}/{len(imagefiles)}] {os.path.basename(f)}', end='\r')
        upscale = (not args.seg)
        output  = model.process_image(f, upscale_result=upscale)
        try:
            output  = model.process_image(f, upscale_result=upscale)
        except Exception as e:
            print(f'Could not process image {os.path.basename(f)}: {e}')
            continue
        
        outf     = os.path.join(outputdir, os.path.basename(f))
        if hasattr(output, 'labelmap'):
            labelmap = output.labelmap
            #save labelmap    
            np.save(outf+'.labelmap.npy', labelmap)

            labelmap_rgba = mplcm.gist_ncar( labelmap / labelmap.max() )
            PIL.Image.fromarray((labelmap_rgba*255).astype('uint8')).save(outf+'.labelmap.png')

            open(outf+'.areas.csv', 'w').write(util.labelmap_to_areas_output(labelmap))

        if hasattr(output, 'boundaries'):
            from src import INBD
            open(outf+'.widths.csv', 'w').write(INBD.boundaries_to_ring_width_output(output.boundaries, scale=model.scale))
            if hasattr(output, 'labelmap'):
                H,W = output.labelmap.shape[-2:]
                open(outf+'.labelmap.svg','w').write(INBD.boundaries_to_svg(output.boundaries, (W,H), model.scale))
        
        if hasattr(output, 'boundary'):
            #segmentation map
            np.save(outf+'.segmentation.npy', output)

            boundaries_normed = np.tanh(output.boundary)/2+0.5
            PIL.Image.fromarray((boundaries_normed*255).astype('uint8')).save(outf+'.segmentation.png')
    print()



def evaluate(args):
    from src import evaluation
    import pickle, json
    annotations = sorted(util.read_splitfile(args.annotations), key=os.path.basename)
    results     = sorted(glob.glob( os.path.join(args.outputs, '*.npy' ) ), key=os.path.basename )
    assert len(annotations) == len(results), [len(annotations), len(results)]
    #TODO: check that the filenames are the same

    print(f'Evaluating {len(results)} results')
    metrics, metrics_raw = evaluation.evaluate_resultfiles(results, annotations)
    print(metrics)
    json.dump(metrics, open(os.path.join(args.outputs, 'metrics.json' ), 'w') )
    pickle.dump( metrics_raw, open(os.path.join(args.outputs, 'metrics.pkl' ), 'wb') )


def update(args):
    '''Update a saved INBD model with new source code'''
    assert os.path.exists(args.model)
    
    model = util.load_model(args.model).eval().requires_grad_(False)
    assert 'INBD' in model.__class__.__name__, NotImplementedError()

    from src import INBD, segmentation
    segmodel     = model.segmentationmodel[0]
    new_segmodel = segmentation.SegmentationModel(
        backbone           = segmodel.backbone_name,
        downsample_factor  = segmodel.scale,
    )
    new_segmodel.load_state_dict(segmodel.state_dict())
    
    new_model = INBD.INBD_Model(
        new_segmodel, 
        backbone           = model.backbone_name,
        wedging_rings      = model.wd_det is not None, 
        angular_density    = model.angular_density,
        concat_radii       = model.concat_radii,
        var_ares           = model.var_ares,
        interpolate_ambiguous = getattr(model, 'interpolate_ambiguous', True), #legacy
    )

    new_model.load_state_dict(model.state_dict())
    outputpath = args.model.replace(".pt.zip", ".update.pt.zip")
    new_model.save(outputpath)





if __name__ == '__main__':
    parser     = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='subcommands', required=True)

    parser_train = subparsers.add_parser('train', help='Train a network')
    parser_train.add_argument('modeltype',                choices=['segmentation', 'INBD', 'MaskRCNN', 'cartesian'], help='Model type')
    parser_train.add_argument('training_images',          type=str, help='Path to a text file containing paths to training images')
    parser_train.add_argument('training_annotations',     type=str, help='Path to a text file containing paths to training annotations')
    parser_train.add_argument('--validation_images',      type=str, help='Path to a text file containing paths to validation images')
    parser_train.add_argument('--validation_annotations', type=str, help='Path to a text file containing paths to validation annotations')
    parser_train.add_argument('--segmentationmodel',      type=str, help='Path to pretrained segmentation model (INBD only)')
    parser_train.add_argument('--epochs',          type=int,   default=100, help='Number of training epochs')
    parser_train.add_argument('--downsample',      type=float, default=4,   help='Resolution downsampling factor')
    parser_train.add_argument('--size',            type=int,   default=512, help='Resolution for the cartesian model')
    parser_train.add_argument('--backbone',        type=str,   default='mobilenet3l',  help='UNet backbone')
    
    parser_train.add_argument('--wd',              type=int,   default=1,    help='Wedging ring detection (WRD)')  #bool
    parser_train.add_argument('--wd_lambda',       type=float, default=0.01, help='WRD loss weight')
    parser_train.add_argument('--radcat',          type=int,   default=1,    help='Concatenate radii as input for the INBD network')  #bool
    parser_train.add_argument('--angular-density', type=float, default=6.28, help='Hyperparameter alpha')
    parser_train.add_argument('--per_epoch_it',    type=int,   default=3,    help='Number of iterations per training epoch')
    parser_train.add_argument('--bd_augment',      type=int,   default=1,    help='Boundary augmentations')
    parser_train.add_argument('--var_ares',        type=int,   default=1,    help='Variable angular resolution')
    parser_train.add_argument('--interpolate_ambiguous', type=int, default=1, help='Interpolate ambiguous boundary points')
    
    
    parser_train.add_argument('--nms',        type=float, default=0.7,   help='Non-Max Suppression threshold (Mask-RCNN)')
    parser_train.add_argument('--mrcnn_acc',  type=int,   default=0,     help='Mask-RCNN: accumulate rings (filled mode)')

    parser_train.add_argument('--lr',         type=float, default=1e-3,  help='Learning rate')
    parser_train.add_argument('--amp',        type=bool,  default=True,  help='Mixed precision training')
    parser_train.add_argument('--val_freq',   type=int,   default=1,     help='Validation frequency (epochs)')

    parser_train.add_argument('--output',     type=str,   default='checkpoints/',   help='Output directory')
    parser_train.add_argument('--suffix',     type=str,   default='',               help='Suffix/description to add to output name')
    parser_train.set_defaults(func=train)



    parser_inf   = subparsers.add_parser('inference', help='Process images with a network')
    parser_inf.add_argument('model',       type=str, help='Path to pretrained model')
    parser_inf.add_argument('images',      type=str, help='Path to a text file containing paths to images')
    parser_inf.add_argument('--output',    type=str, default='inference/', help='Output directory')
    parser_inf.add_argument('--suffix',    type=str, default='',           help='Suffix/description to add to output name')
    parser_inf.add_argument('--seg',       type=bool,default=False,        help='Save only segmentation output')
    parser_inf.set_defaults(func=inference)



    parser_eval  = subparsers.add_parser('evaluate',  help='Evaluate a network')
    parser_eval.add_argument('outputs',     type=str, help='Path to a folder containing inference outputs')
    parser_eval.add_argument('annotations', type=str, help='Path to a text file containing paths to annotations')
    parser_eval.set_defaults(func=evaluate)

    parser_up = subparsers.add_parser("update", help="Update a saved model with new source code")
    parser_up.set_defaults(func=update)
    parser_up.add_argument("model", type=str, help="Path to model")

    args = parser.parse_args(sys.argv[1:] or ['--help'])
    args.func(args)

    print('Done')
