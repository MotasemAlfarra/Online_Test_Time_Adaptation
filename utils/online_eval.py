import torch 
import math
from time import time
from tqdm import tqdm
from utils.memo_utils.memo_transforms import te_transforms_inc
from utils.memo_utils.third_party_memo import imagenet_r_mask

def delayed_eval_online(model, data_loader, eta, device, dataset_name, single_model=False):
    def compute_time_and_output(model, images, device, dataset_name):
        images = images.to(device)
        # Calculating the consumed time for each forward pass
        torch.cuda.synchronize()
        start_time = time()
        output = model(images)
        torch.cuda.synchronize()
        consumed_time = time()-start_time
        if dataset_name == 'imagenetr':
            output = output[:, imagenet_r_mask]
        return output.argmax(1), consumed_time

    num_samples, num_correct = 0, 0
    delay = 0
    
    with torch.no_grad():        
        for _, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)
            if delay == 0: # Compute the output and delay for this batch
                output, time_for_tte = compute_time_and_output(model, images, device, dataset_name)
                _, time_for_base = compute_time_and_output(model.model, images, device, dataset_name)
                ratio = time_for_tte/time_for_base
                delay += max(0, math.ceil(ratio * eta) - 1)
            else: # Previous expensive algorithm is still running
                if single_model:
                    output = torch.randint(0, len(data_loader.dataset.classes), (images.shape[0],), device=device)
                else:
                    output, _ = compute_time_and_output(model.model, images, device, dataset_name)
                delay -= 1
            num_correct += output.eq(labels).sum().item()
            num_samples += images.shape[0]

    return num_correct/num_samples, model


def delayed_eval_online_memo(model, dataset, eta, device, dataset_name, **kwargs):
    def compute_time_and_output(model, images, device, dataset_name):
        # images = images.to(device)
        # Calculating the consumed time for each forward pass
        start_time = time()
        output = model(images)
        if dataset_name == 'imagenetr':
            output = output[:, imagenet_r_mask]        
        consumed_time = time()-start_time
        return output.argmax(1), consumed_time

    num_samples, num_correct = 0, 0
    delay = 0
    # with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        images, labels = dataset[i]
            
        if delay == 0: # Compute the output and delay for this batch
            output, time_for_tte = compute_time_and_output(model, images, device, dataset_name)
            _, time_for_base = compute_time_and_output(model.model, te_transforms_inc(images).unsqueeze(0).cuda(), device, dataset_name)
            delay += max(0, math.ceil(eta * time_for_tte/time_for_base) - 1)
        else: # Previous expensive algorithm is still running
            output = model.model(te_transforms_inc(images).unsqueeze(0).cuda())
            if dataset_name == 'imagenetr':
                output = output[:, imagenet_r_mask]
            output = output.argmax(1)
            delay -= 1
        num_correct += output.eq(labels).sum().item()
        # num_samples += images.shape[0]  
        num_samples += 1
    return num_correct/num_samples, model
