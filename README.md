<!-- # Online_Test_Time_Adaptation -->
# Revisiting Test Time Adaptation Under Online Evaluation

![plot](./pipeline.png)

This benchmark is a step towards standardizing the evaluation of Test Time Adaptation (TTA) methods. We have implementations of 14 different TTA methods from the literature. 

- Source / Non-Adapted: Pretrained ResNet-50 on ImageNet where the weights are loaded from `torchvision`. 
- AdaBN : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- SHOT and SHOT-IM : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- TENT :  [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- BN Adaptation : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- SAR :  [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- CoTTA : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- TTAC-NQ : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- ETA and EATA : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness) 
- MEMO :  [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- DDA : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- PL : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- LAME :  [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)

We evaluate all considered methods with varying the rate in which the stream of data is revealing new batches for the TTA method.

## Environment Installation
To use our code, first you might need to install our environment through running:

```
conda env install -f environment.yml
```

## Datasets used for Evaluation
Our results are reported on 3 different datasets: ImageNet-C, ImageNet-R, and ImageNet-3DCC. 
All datasets are publicly available and can be downloaded from their corresponding repositories. 
- ImageNet-C: [here](https://github.com/hendrycks/robustness)
- ImageNet-R: [here](https://github.com/hendrycks/imagenet-r)
- ImageNet-3DCC: [here](https://github.com/EPFL-VILAB/3DCommonCorruptions)

## Reproducing Results in Our Paper





