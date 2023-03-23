import torch 
import math
from time import time
from tqdm import tqdm
from utils.memo_transforms import te_transforms_inc
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