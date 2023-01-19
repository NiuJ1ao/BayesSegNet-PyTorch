# A PyTorch implementation of Bayesian SegNet

## Requirements
TBD

## Data
The models are tested on the CamVid dataset from [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial).

## Usage
```
python train.py --model <SegNet|BayesSegNet> --data-path <Path to CamVid dataset> --device <cpu|cuda:x>
```

## Performance on CamVid dataset
| Model | Class Avg. | mIoU |
|:------|:-----------|:-----|
| SegNet | 73.76% | 62.74% |
| Bayes SegNet | 79.82% | 63.94% |

