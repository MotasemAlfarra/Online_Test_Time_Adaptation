import os
import torch
from utils.argparse import get_args
from models.resnets import _all_models 
from utils.online_eval import delayed_eval_online, delayed_eval_online_memo
from utils.dataloader import get_dataloader, get_cp
from tta_methods import _all_methods

def main(args):
    
    if args.use_wandb:
        import wandb
        wandb.init(project="tta", config=args)
    print(args)
    
    # Initializing the model
    model = _all_models[args.arch](pretrained=True, progress=True).to(args.device)

    # Putting the model into a wrapper
    tta_method = _all_methods[args.method](model, args)

    # Getting the corrupted data that we need to evaluate the model at
    all_corruptions = get_cp(args)
    print(all_corruptions)

    for corruption in all_corruptions: 
         
        args.corruption = corruption
        print("loading "+args.corruption+" corruption ...")
        if args.single_model:
            print('Performing single model evaluation')
        corrupted_dataloader = get_dataloader(args)

        # MEMO has a different evaluation function due to their implementation
        delayed_func = delayed_eval_online_memo if args.method == 'memo' else delayed_eval_online

        # Evaluating the model
        adjusted_acc, tta_method = delayed_func(tta_method, corrupted_dataloader, delay=args.eta, \
                                                device=args.device, dataset_name=args.dataset, single_model=args.single_model)

        # logger.info(args.corruption)
        print(f"Under shift type {args.corruption} After {args.method} Top-1 Adjusted Accuracy: {adjusted_acc*100:.5f}")
        print(f"Under shift type {args.corruption} After {args.method} Top-1 Error Rate: {100-adjusted_acc*100:.5f}")
        print(f'Finished {args.method} on {args.corruption} with level {args.level}, Adjusted Error Rate: {100-adjusted_acc*100:.5f}, eta: {args.eta}')
        
        with open(os.path.join(args.output, '{}.txt'.format(args.corruption)), 'w') as f:
            f.write('eta {}'.format(args.eta))
            f.write('\n')
            f.write('error_rate {}'.format(100-100*adjusted_acc))
    
    if args.use_wandb:
        wandb.log({'adjusted_acc': adjusted_acc, 'error_rate': 100-100*adjusted_acc, 'eta': args.eta})
    return

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    main(args)