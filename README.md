# Mixup-Implementaion
Implementation for ["mixup: BEYOND EMPIRICAL RISK MINIMIZATION"](https://arxiv.org/abs/1710.09412)

### To train ResNet18 with mixup on CIFAR10:
```!
python Mixup-Implementaion/src/main.py --epochs 2 --batch-size 128 --learning-rate 0.1 --save-path "[path to save the model and tensorboard logs]"      
```

The resnnet18 architecture is the cifar10 variant of resnet18. I borrowed the architecture code from the official implementation.  

| Model            | epochs        | cifar10 val acc|
| -------------    | ------------- |  ------------- |
| Resnet18         |  200          |      94.61     |
| Resnet18_mixup   |  200          |     95.99      |


<img width="600" alt="mixup graph" src="https://user-images.githubusercontent.com/37993690/130130616-c8de87c0-3dd4-418d-8f59-fb70b1d3eabc.png">




