<!-- # Online_Test_Time_Adaptation -->
# Revisiting Test Time Adaptation Under Online Evaluation

![plot](./pipeline.png)

This benchmark is a step towards standardizing the evaluation of Test Time Adaptation (TTA) methods. We have implementations of 14 different TTA methods from the literature. 

- Source / Non-Adapted: Pretrained ResNet-50 on ImageNet where the weights are loaded from `torchvision`. 
- AdaBN : [(paper)](https://arxiv.org/abs/1603.04779), [(code)](https://github.com/erlendd/ddan)
- SHOT and SHOT-IM : [(paper)](https://arxiv.org/abs/2002.08546), [(code)](https://github.com/fiveai/LAME/blob/master/src/adaptation/shot.py)
- TENT :  [(paper)](https://openreview.net/pdf?id=uXl3bZLkr3c), [(code)](https://github.com/DequanWang/tent)
- BN Adaptation : [(paper)](https://arxiv.org/pdf/2006.16971v1.pdf), [(code)](https://github.com/bethgelab/robustness/tree/main/examples/batchnorm)
- SAR :  [(paper)](https://openreview.net/forum?id=g2YraF75Tj), [(code)](https://github.com/mr-eggplant/SAR)
- CoTTA : [(paper)](https://arxiv.org/abs/2203.13591), [(code)](https://github.com/qinenergy/cotta)
- TTAC-NQ : [(paper)](https://openreview.net/forum?id=g2YraF75Tj), [(code)](https://github.com/Gorilla-Lab-SCUT/TTAC)
- ETA and EATA : [(paper)](https://arxiv.org/abs/2204.02610), [(code)](https://github.com/mr-eggplant/EATA) 
- MEMO :  [(paper)](https://arxiv.org/abs/2110.09506), [(code)](https://github.com/zhangmarvin/memo)
- DDA : [(paper)](https://arxiv.org/abs/2207.03442), [(code)](https://github.com/shiyegao/DDA)
- PL : [(paper)](https://github.com/hendrycks/robustness), [(code)](https://github.com/hendrycks/robustness)
- LAME :  [(paper)](https://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.pdf), [(code)](https://github.com/fiveai/LAME)

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





