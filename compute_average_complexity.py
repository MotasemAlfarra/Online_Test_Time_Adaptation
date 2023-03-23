import os
import torch
from utils.argparse import get_args
from models.resnets import resnet50 
from utils.average_complexity_utils import compute_time, compute_time_memo
from utils.dataloader import initial_data
from tte_methods import _all_methods
from statistics import median

def main(args):
        
    # Initializing the model
    if args.arch == 'resnet50':
        model = resnet50(pretrained=True, progress=True).to(args.device)

    # Putting the model into a wrapper
    ttt_method = _all_methods[args.method](model, args)
    delay_list = []

    for _ in range(args.runs):
        # Getting random data to calculate time
        initial_dataloader = initial_data(args)
        # Calculating the delay for a given method
        tte_method_time = compute_time(ttt_method, initial_dataloader, args.device) if args.method != 'memo' else compute_time_memo(ttt_method, initial_dataloader, args.device)
        # logger.info("The average time for {} forward pass with batch size 1 is {}seconds".format(args.arch + '_' + args.method, tte_method_time))
        base_model_time = compute_time(model, initial_dataloader, args.device) if args.method != 'memo' else compute_time_memo(model, initial_dataloader, args.device, True)
        # logger.info("The average time for {} forward pass with batch size 1 is {} seconds".format(args.arch, base_model_time))

        delay = tte_method_time / base_model_time
        delay_list. append(delay)
    
    print("The delays are: ", delay_list)
    median_delay = median(delay_list)
    print("The median delays is: ", median_delay)
    
    path = os.path.join('average_delays')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, '{}.txt'.format(args.method)), 'w') as f:
        string_delay_list = '[' + ','.join([str(delay) for delay in delay_list]) + ']'
        f.write(string_delay_list)
        f.write('\n')
        f.write('{}'.format(median_delay))
    
    return

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    main(args)

