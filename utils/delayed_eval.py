import torch 
import math
from time import time
from tqdm import tqdm
from tte_methods.memo_transforms import te_transforms_inc
from utils.third_party_memo import imagenet_r_mask

def compute_time(model, val_loader, device):
    start_time = time()
    consumed_time = 0.0
    with torch.no_grad():
        for _, (images, _) in enumerate(tqdm(val_loader)):
            assert images.shape[0] == 1, \
            "Batch size here should be one for fair comparison"
            images = images.to(device)
            # Calculating the consumed time for each forward pass
            start_time = time()
            _ = model(images)
            consumed_time += time()-start_time
    return consumed_time/len(val_loader)


def delayed_eval(model, data_loader, delay, device, dataset_name, **kwargs):
    num_samples, num_correct = 0, 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader)):

            images, labels = images.to(device), labels.to(device)
            
            if i%delay == 0: #Finished the previous expensive computation
                output = model(images)
            else:            #Previous expensive algorithm is still running
                output = model.model(images)
            if dataset_name == 'imagenetr':
                output = output[:, imagenet_r_mask]
            output = output.argmax(1)
            num_correct += output.eq(labels).sum().item()
            num_samples += images.shape[0]
            
    return num_correct/num_samples, model


def delayed_eval_online(model, data_loader, delay, device, dataset_name, single_model=False):
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
    stream_speed = delay
    delay = 0
    
    with torch.no_grad():        
        for _, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)
            if delay == 0: # Compute the output and delay for this batch
                output, time_for_tte = compute_time_and_output(model, images, device, dataset_name)
                _, time_for_base = compute_time_and_output(model.model, images, device, dataset_name)
                ratio = time_for_tte/time_for_base
                delay += max(0, math.ceil(ratio * stream_speed) - 1)
            else: # Previous expensive algorithm is still running
                if single_model:
                    output = torch.randint(0, len(data_loader.dataset.classes), (images.shape[0],), device=device)
                else:
                    output, _ = compute_time_and_output(model.model, images, device, dataset_name)
                delay -= 1
            num_correct += output.eq(labels).sum().item()
            num_samples += images.shape[0]

    return num_correct/num_samples, model


def compute_time_memo(model, dataset, device, with_transforms = False):
    start_time = time()
    consumed_time = 0.0
    # with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        images, _ = dataset[i]
        # Calculating the consumed time for each forward pass
        start_time = time()
        if with_transforms: 
            images = te_transforms_inc(images).unsqueeze(0).cuda()
        _ = model(images)
        consumed_time += time()-start_time
    return consumed_time/len(dataset)


def delayed_eval_memo(model, dataset, delay, device,  dataset_name, **kwargs):
    num_samples, num_correct = 0, 0
    for i in tqdm(range(len(dataset))):
        images, labels = dataset[i]
        # if labels != 0:
        #     continue

        if i%delay == 0: #Finished the previous expensive computation
            output = model(images)
        else:            #Previous expensive algorithm is still running
            with torch.no_grad():
                output = model.model(te_transforms_inc(images).unsqueeze(0).cuda())
        if dataset_name == 'imagenetr':
            output = output[:, imagenet_r_mask]
        output = output.argmax(1)     
        num_correct += output.eq(labels).sum().item()
        num_samples += 1

    return num_correct/num_samples, model


def delayed_eval_online_memo(model, dataset, delay, device, dataset_name, **kwargs):
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
            delay += max(0, math.ceil(time_for_tte/time_for_base) - 1)
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
