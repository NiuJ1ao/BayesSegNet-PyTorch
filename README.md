# A PyTorch implementation of Bayesian SegNet
This repository includes the PyTorch implementation of SegNet and Bayes SegNet.

## Requirements
TBD

## Data
The models are tested on the CamVid dataset from [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial).

## Train Bayes SegNet
```
python train.py --data-path <Path to CamVid dataset> --device <cpu|cuda:x>
```

## Performance on CamVid dataset
| Model | Class Avg. | mIoU |
|:------|:-----------|:-----|
| SegNet | 70.30% | 53.30% |
| Bayes SegNet | 76.78% | 60.72% |

