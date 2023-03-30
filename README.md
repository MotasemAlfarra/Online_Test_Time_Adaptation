# Online_Test_Time_Adaptation
Revisiting Test Time Adaptation Under Online Evaluation

This benchmark is a step towards standardizing the evaluation of Test Time Adaptation (TTA) methods. We have implementations of 14 different TTA methods from the literature:

- Source / Non-Adapted
- AdaBN : 
- SHOT and SHOT-IM :
- TENT : 
- BN Adaptation :
- SAR : 
- CoTTA :
- TTAC-NQ :
- ETA and EATA : 
- MEMO : 
- DDA :
- PL :
- LAME : 

We evaluate all considered methods with varying the rate in which the stream of data is revealing new batches for the TTA method.

## Environment Installation
To use our code, first you might need to install our environment through running:

conda env install -f environment.yml

## Reproducing Results
Our results are reported on 3 different benchmarks. For ImageNet-C, you might need to download the dataset from (here).
For ImageNet-3DCC, the dataset is available (here). Last, for ImageNet-R that dataset can be downloaded from (here).




