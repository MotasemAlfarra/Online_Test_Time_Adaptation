import os
import argparse
import math 

_all_methods = ['basic', 'tent', 'eta', 'eata', 'cotta', 'ttac_nq', 'memo', 'adabn', 'shot', 'shotim', 'lame', 'bn_adaptation', 'pl', 'adacontrast', 'sar', 'dda']

_common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

_3dcc_corruptions = ['bit_error', 'color_quant', 'far_focus', 'flash', 'fog_3d', 'h265_abr', 'h265_crf', \
                        'iso_noise', 'low_light', 'near_focus', 'xy_motion_blur', 'z_motion_blur']

_all_corruptions = ['all', 'all_ordered', *_common_corruptions, *_3dcc_corruptions]

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')
    parser.add_argument('--runs', default=10, type=int, help='how many runs to compute median delay')
    parser.add_argument('--delay', default='online', choices=['average', 'online', 'reproduce'], type=str, help='delay in case you want to test the method on other delays')
    parser.add_argument('--delay_value', default=1, type=float, help='delay value in case you want to test the method on other delays, default is 1 to reproduce the results in the papers')
    
    parser.add_argument('--single_model', default=False, action='store_true', help='evaluate assuming a single model can be deployed')
    
    parser.add_argument('--use_wandb', default=False, action='store_true', help='use wandb for logging')
    
    parser.add_argument('--initial_data_size', default=500, type=int, help='initial data size to compute delay (default: 500)')
    parser.add_argument('--spatial_dim_size', default=224, type=int, help='image spatial dimension (default: 224)')
    parser.add_argument('--initial_batch_size', default=1, type=int, help='batch size for delay computation (default: 1)')
    
    # path of data, output dirspatial_dim_size
    parser.add_argument('--dataset', default='imagenetc', help='imagenetc or generated_imagenetc', choices=['imagenetc', 'generated_imagenetc', 'imagenet3dcc', 'imagenetr'])
    parser.add_argument('--imagenet_path', default='/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg/', help='path to imagenetc dataset')
    parser.add_argument('--imagenetc_path', default='/ibex/ai/reference/CV/ImageNet-C', help='path to imagenetc dataset')
    parser.add_argument('--generated_imagenetc_path', default='./data/ImageNet/Generated_ImageNet-C', help='path to generated imagenetc dataset')
    parser.add_argument('--imagenetr_path', default='/ibex/ai/reference/CV/imagenet-r', help='path to ImageNet-R dataset')
    parser.add_argument('--imagenet3dcc_path', default='/ibex/ai/reference/CV/ImageNet-3DCC', # default='/ibex/scratch/projects/c2138/juan/ImageNet-3DCC', 
                        help='path to ImageNet-3DCC dataset') # provisionally!!! <<----

    parser.add_argument('--finetune_val', default=False, action='store_true', help='add imagenet val at begining of experiment')
    parser.add_argument('--test_val', default=False, action='store_true', help='add imagenet val at end of experiment')

    parser.add_argument('--output', default='./output/', help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu.')

    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--shuffle', default=True, type=bool, help='if shuffle the test set.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='defocus_blur', type=str, choices=_all_corruptions, help='corruption type of test(val) set.')

    # Which TTT method to evaluate
    parser.add_argument('--method', default='basic', type=str, choices=_all_methods, help='the method to use for TTT')
    
    # optimizer related parameters
    parser.add_argument('--optimizer', type=str, default='sgd', help='Which optimizer to use')
    parser.add_argument('--steps', type=int, default=1, help='How many optimization steps to perform')

    # EATA and ETA specific hyperparameters
    parser.add_argument('--fisher_size', type=float, default=2000, help='Number of samples on which fishers are computed over')
    parser.add_argument('--fisher_alpha', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    
    # LAME specific hyperparameters
    parser.add_argument('--affinity_type', type=str, default='knn', help='Type of affinity to use for LAME')
    parser.add_argument('--knn', type=int, default=5, help='Number of nearest neighbors to use for LAME')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian affinity')
    
    # PL specific parameters
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for PL, taken from LAME')
    
    # SHOT/SHOT-IM specific parameters, maybe add it to other methos? Optimal LR is 0.01, try also 0.001
    parser.add_argument('--update_bn_only', action='store_true', help='Update backbone parameters by default, only update BN params (like tent) is required')
    parser.add_argument('--beta_clustering_loss', default=0.1, type=float, help='beta for clustering loss, if 0 SHOT-IM is used')
    
    ######## Not used args but could be useful for future
    parser.add_argument('--arch', default='resnet50', type=str, help='the default model architecture') # not used for now   
    parser.add_argument('--data', default='ImageNet', help='path to dataset') #not used for now
    # overall experimental settings
    # 'cotinual' means the model parameters will never be reset, also called online adaptation; 
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--exp_type', default='each_shift_reset', type=str, help='continual or each_shift_reset')# not used for now

    # added for Memo
    parser.add_argument('--with_transforms', dest='transforms', action='store_true', help='applying default transforms of the dataset')
    parser.add_argument('--no_transforms', dest='transforms', action='store_false', help='applying default transforms of the dataset')
    parser.set_defaults(transforms = True)
    parser.add_argument('--return_dataset', dest='r_dataset', action='store_true', help='return dataset instead of dataloader useful if want to interate one-by-one')
    parser.add_argument('--no_return_dataset', dest='r_dataset', action='store_false', help='return dataset instead of dataloader useful if want to interate one-by-one')
    parser.set_defaults(r_dataset = False)
    args = parser.parse_args()

    
    args.output = os.path.join(args.output, args.dataset, args.method, args.method + '_' + args.delay)
    # args.output = os.path.join(args.output, args.dataset, args.method + '_' + str(args.delay))
    # args.output = os.path.join(args.output, args.corruption + '_' + str(args.level))
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    return args