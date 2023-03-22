import os
import math
import torch
from utils.argparse import get_args
from models.resnets import _all_models 
from utils.delayed_eval import compute_time, delayed_eval, delayed_eval_memo, delayed_eval_online, delayed_eval_online_memo
from utils.dataloader import get_dataloader, get_cp
# from utils.get_logger import get_logger
from tte_methods import _all_methods
# import timm
common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', \
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', \
                        'elastic_transform', 'pixelate', 'jpeg_compression']
_3dcc_corruptions = ['bit_error', 'color_quant', 'far_focus', 'flash', 'fog_3d', 'h265_abr', 'h265_crf', \
                        'iso_noise', 'low_light', 'near_focus', 'xy_motion_blur', 'z_motion_blur']

def main(args):
    
    if args.use_wandb:
        import wandb
        wandb.init(project="tte", config=args)
    # logger = get_logger(name="project", output_directory=args.output, log_name="log.txt", debug=False)
    print(args)
    
    # Initializing the model
    model = _all_models[args.arch](pretrained=True, progress=True).to(args.device)

    # Putting the model into a wrapper
    ttt_method = _all_methods[args.method](model, args)

    if args.delay == 'average':
        path = os.path.join('average_delays','{}.txt'.format(args.method))
        
        if not os.path.isfile(path):
            print('The average delay for {} is not computed'.format(args.method))
            return
        
        with open(path) as f:
            lines = f.readlines()

        delay = max(1,math.ceil(float(lines[1])))
    else:
        delay = args.delay_value
    
    print("The delay is {}".format(delay))

    

    # Getting the corrupted data that we need to evaluate the model at
    corrs_to_use = common_corruptions if args.dataset in ['imagenetc', 'generated_imagenetc'] else _3dcc_corruptions
    all_corruptions = get_cp(args.dataset) if args.corruption == 'all' else corrs_to_use if args.corruption == 'all_ordered' else [args.corruption]
    
    if args.finetune_val:
        all_corruptions = ['val', *all_corruptions]
    if args.test_val:
        all_corruptions = [*all_corruptions,'val']
    
    if args.dataset == 'imagenetr':
        all_corruptions = ['imagenetr']
    for corruption in all_corruptions: 
         
        args.corruption= corruption
        print("loading "+args.corruption+" corruption ...")
        if args.single_model:
            print('Performing single model evaluation')
        corrupted_dataloader = get_dataloader(args)
        # if args.exp_type == "each_shift_reset":
            # model reset
            # ttt_method = _all_methods[args.method](model, args) 
        # Calculating the delayed evaluation
        if args.delay == 'online':
            delayed_func = delayed_eval_online_memo if args.method == 'memo' else delayed_eval_online
        else: 
            # delayed_evaluation = delayed_eval_online if args.delay=='online' else delayed_eval
            delayed_func = delayed_eval_memo if args.method == 'memo' else delayed_eval

        adjusted_acc, ttt_method = delayed_func(ttt_method, corrupted_dataloader, delay=delay, device=args.device, dataset_name=args.dataset, single_model=args.single_model)

        # logger.info(args.corruption)
        print(f"Under shift type {args.corruption} After {args.method} Top-1 Adjusted Accuracy: {adjusted_acc*100:.5f}")
        print(f"Under shift type {args.corruption} After {args.method} Top-1 Error Rate: {100-adjusted_acc*100:.5f}")
        print(f'Finished {args.method} on {args.corruption} with level {args.level}, Adjusted Error Rate: {100-adjusted_acc*100:.5f}, delay: {delay}')
        
        with open(os.path.join(args.output, '{}.txt'.format(args.corruption)), 'w') as f:
            f.write('delay {}'.format(delay))
            f.write('\n')
            f.write('error_rate {}'.format(100-100*adjusted_acc))
    
    if args.use_wandb:
        wandb.log({'adjusted_acc': adjusted_acc, 'error_rate': 100-100*adjusted_acc, 'delay': delay})
    return

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    main(args)