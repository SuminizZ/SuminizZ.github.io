---
layout: post
title : "[Paper Review & Implementation] Going deeper with convolutions (GoogleNet/Inception Net v1, 2014)"
img: papers/cv/inception_v1.png
categories: [papers-cv]  
tag : [Paper Review, CNN, GoogleNet, Inception Net v1]
toc : true
toc_sticky : true
---

## **Outlines** 
- [References](#references)
- [Inception V1 Architecture](#inception-v1-architecture)
- [Implementation with PyTorch](#implementation-with-pytorch)
- [Model Summary](#model-summary)
- [Forward Pass](#forward-pass)

<br/>


## **References**
- [Going deeper with convolutions, Christian Szegedy (2014)](https://arxiv.org/abs/1409.4842){:target="_blank"}
- [https://hyukppen.modoo.at/?link=5db82s6p](https://hyukppen.modoo.at/?link=5db82s6p){:target="_blank"}

<br/>

## **Inception V1 Architecture**

<br/>
<p align="center"><img src="https://github.com/SuminizZ/Physics/assets/92680829/86b9a02c-93b7-4c52-a71f-465a3e99a431" width="670px"></p>

<br/>
<p align="center"><img width="600px" alt="image" src="https://github.com/SuminizZ/Physics/assets/92680829/b9c4f89d-e0e6-4d2b-819a-5725cb549c51"></p>

<br/>

1. improves computational efficiency 
- reduce dimension of parameters by adding extra 1x1 conv layer before 3x3 and 5x5 conv layers 
- take global average pooling before entering into fc layer 

2.  Use different sizes of filters to perform convolution on a single input and concatenate them into one output (output sizes adjusted with padding)
- combination
- recorded lowest error at ImageNet classifcation

3. mitigate gradient vanishing problem with auxiliary classifiers
- increase the gradient signal from intermediate or lower layers but with regularization factor 0.3 (loss = final_out_loss + 0.3*(aux1_loss + aux2_loss)
- -> turns out this is not the case. (these branches not help reflecting low-level features) Instead, they work as regularizers with batch noramlization applied

<br/>

<img src="https://github.com/SuminizZ/Physics/assets/92680829/29509567-5368-41f9-b847-538a95251520">

<br/>

## **Implementation with PyTorch**

<br/>

```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append("/content/drive/MyDrive/Legend13")
import torch 
import torch.nn as nn
!pip install torchinfo
from torchinfo import summary
```

<br/>

    Mounted at /content/drive
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting torchinfo
      Downloading torchinfo-1.8.0-py3-none-any.whl (23 kB)
    Installing collected packages: torchinfo
    Successfully installed torchinfo-1.8.0


<br/>


```python
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, F, size, stride, padding=0):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channel, F, size, stride, padding=padding),
                                   nn.BatchNorm2d(F),
                                   nn.ReLU())

    def forward(self, x):
        out = self.layer(x)
        return out 


class Inception(nn.Module):
    def __init__(self, in_channel, Fs, final_F):
        super().__init__()
        self.final_F = final_F
        c1_F, c3_red_F, c3_F, c5_red_F, c5_F, poolproj_F = Fs
        self.conv1x1 = BasicConv2d(in_channel, c1_F, 1, 1, padding=0)
        self.conv3x3 = nn.Sequential(BasicConv2d(in_channel, c3_red_F, 1, 1, padding=0),
                                     BasicConv2d(c3_red_F, c3_F, 3, 1, padding=1))
        self.conv5x5 = nn.Sequential(BasicConv2d(in_channel, c5_red_F, 1, 1, padding=0),
                                     BasicConv2d(c5_red_F, c5_F, 5, 1, padding=2))
        self.maxpool_conv1x1 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1),
                                             BasicConv2d(in_channel, poolproj_F, 1, 1, padding=0))

    def forward(self, x):
        x_concat = [self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.maxpool_conv1x1(x)]
        x_concat = torch.cat(x_concat, 1)   # N,F,H,W
        # assert x_concat.shape[1] == self.final_F
        
        return x_concat


class AuxOut(nn.Module):
    def __init__(self, in_channel, p, num_classes):
        super().__init__()
        self.avgpool_conv = nn.Sequential(nn.AvgPool2d(5, 3, padding=0),
                                          BasicConv2d(in_channel, 128, 1, 1, padding=0))
        self.fc = nn.Sequential(nn.Linear(2048, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=p),
                                nn.Linear(1024, num_classes))
        
    def forward(self, x):
        x = self.avgpool_conv(x)
        x = torch.flatten(x,1)
        out = self.fc(x)
        return out 


class Inception_V1(nn.Module):
    def __init__(self, init_weights=True, p=0.5, use_aux=True, in_channel=3, num_classes=1000):
        super().__init__()
        self.use_aux = use_aux

        self.conv1 = BasicConv2d(in_channel, 64, 7, 2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)

        self.conv2a = BasicConv2d(64, 64, 1, 1, padding=0)
        self.conv2b = BasicConv2d(64, 192, 3, 1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, padding=1)

        Fs = (64, 96, 128, 16, 32, 32)
        self.inception3a = Inception(192, Fs, 256)
        Fs = (128, 128, 192, 32, 96, 64)
        self.inception3b = Inception(256, Fs, 480)
        self.maxpool3 = nn.MaxPool2d(3, 2, padding=1)

        Fs = (192, 96, 208, 16, 48, 64)
        self.inception4a = Inception(480, Fs, 512)
        Fs = (160, 112, 224, 24, 64, 64)
        self.inception4b = Inception(512, Fs, 512)
        Fs = (128, 128, 256, 24, 64, 64)
        self.inception4c = Inception(512, Fs, 512)
        Fs = (112, 144, 288, 32, 64, 64)
        self.inception4d = Inception(512, Fs, 528)
        Fs = (256, 160, 320, 32, 128, 128)
        self.inception4e = Inception(528, Fs, 832)
        self.maxpool4 = nn.MaxPool2d(3, 2, padding=1)

        Fs = (256, 160, 320, 32, 128, 128)
        self.inception5a = Inception(832, Fs, 832)
        Fs = (384, 192, 384, 48, 128, 128)
        self.inception5b = Inception(832, Fs, 1024)

        if use_aux:
            self.aux1 = AuxOut(512, p, num_classes)
            self.aux2 = AuxOut(528, p, num_classes)
        else:
            self.aux1, self.aux2 = None, None
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=p)
        self.classifier = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                    nn.init.trunc_normal_(m.weight, 0, 1e-2, a=-2, b=2)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux1 is not None:
            aux1_out = self.aux1(x)
        else: aux1_out = None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.training and self.aux2 is not None:
            aux2_out = self.aux2(x)
        else: aux2_out = None

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.classifier(x)

        return aux1_out, aux2_out, out
```

<br/>

## **Model Summary**

<br/>


```python
model = Inception_V1()
summary(model, input_size=(2, 3, 224, 224), device='cpu')
```

<br/>



    ===============================================================================================
    Layer (type:depth-idx)                        Output Shape              Param #
    ===============================================================================================
    Inception_V1                                  --                        6,380,240
    ├─BasicConv2d: 1-1                            [2, 64, 112, 112]         --
    │    └─Sequential: 2-1                        [2, 64, 112, 112]         --
    │    │    └─Conv2d: 3-1                       [2, 64, 112, 112]         9,472
    │    │    └─BatchNorm2d: 3-2                  [2, 64, 112, 112]         128
    │    │    └─ReLU: 3-3                         [2, 64, 112, 112]         --
    ├─MaxPool2d: 1-2                              [2, 64, 56, 56]           --
    ├─BasicConv2d: 1-3                            [2, 64, 56, 56]           --
    │    └─Sequential: 2-2                        [2, 64, 56, 56]           --
    │    │    └─Conv2d: 3-4                       [2, 64, 56, 56]           4,160
    │    │    └─BatchNorm2d: 3-5                  [2, 64, 56, 56]           128
    │    │    └─ReLU: 3-6                         [2, 64, 56, 56]           --
    ├─BasicConv2d: 1-4                            [2, 192, 56, 56]          --
    │    └─Sequential: 2-3                        [2, 192, 56, 56]          --
    │    │    └─Conv2d: 3-7                       [2, 192, 56, 56]          110,784
    │    │    └─BatchNorm2d: 3-8                  [2, 192, 56, 56]          384
    │    │    └─ReLU: 3-9                         [2, 192, 56, 56]          --
    ├─MaxPool2d: 1-5                              [2, 192, 28, 28]          --
    ├─Inception: 1-6                              [2, 256, 28, 28]          --
    │    └─BasicConv2d: 2-4                       [2, 64, 28, 28]           --
    │    │    └─Sequential: 3-10                  [2, 64, 28, 28]           12,480
    │    └─Sequential: 2-5                        [2, 128, 28, 28]          --
    │    │    └─BasicConv2d: 3-11                 [2, 96, 28, 28]           18,720
    │    │    └─BasicConv2d: 3-12                 [2, 128, 28, 28]          110,976
    │    └─Sequential: 2-6                        [2, 32, 28, 28]           --
    │    │    └─BasicConv2d: 3-13                 [2, 16, 28, 28]           3,120
    │    │    └─BasicConv2d: 3-14                 [2, 32, 28, 28]           12,896
    │    └─Sequential: 2-7                        [2, 32, 28, 28]           --
    │    │    └─MaxPool2d: 3-15                   [2, 192, 28, 28]          --
    │    │    └─BasicConv2d: 3-16                 [2, 32, 28, 28]           6,240
    ├─Inception: 1-7                              [2, 480, 28, 28]          --
    │    └─BasicConv2d: 2-8                       [2, 128, 28, 28]          --
    │    │    └─Sequential: 3-17                  [2, 128, 28, 28]          33,152
    │    └─Sequential: 2-9                        [2, 192, 28, 28]          --
    │    │    └─BasicConv2d: 3-18                 [2, 128, 28, 28]          33,152
    │    │    └─BasicConv2d: 3-19                 [2, 192, 28, 28]          221,760
    │    └─Sequential: 2-10                       [2, 96, 28, 28]           --
    │    │    └─BasicConv2d: 3-20                 [2, 32, 28, 28]           8,288
    │    │    └─BasicConv2d: 3-21                 [2, 96, 28, 28]           77,088
    │    └─Sequential: 2-11                       [2, 64, 28, 28]           --
    │    │    └─MaxPool2d: 3-22                   [2, 256, 28, 28]          --
    │    │    └─BasicConv2d: 3-23                 [2, 64, 28, 28]           16,576
    ├─MaxPool2d: 1-8                              [2, 480, 14, 14]          --
    ├─Inception: 1-9                              [2, 512, 14, 14]          --
    │    └─BasicConv2d: 2-12                      [2, 192, 14, 14]          --
    │    │    └─Sequential: 3-24                  [2, 192, 14, 14]          92,736
    │    └─Sequential: 2-13                       [2, 208, 14, 14]          --
    │    │    └─BasicConv2d: 3-25                 [2, 96, 14, 14]           46,368
    │    │    └─BasicConv2d: 3-26                 [2, 208, 14, 14]          180,336
    │    └─Sequential: 2-14                       [2, 48, 14, 14]           --
    │    │    └─BasicConv2d: 3-27                 [2, 16, 14, 14]           7,728
    │    │    └─BasicConv2d: 3-28                 [2, 48, 14, 14]           19,344
    │    └─Sequential: 2-15                       [2, 64, 14, 14]           --
    │    │    └─MaxPool2d: 3-29                   [2, 480, 14, 14]          --
    │    │    └─BasicConv2d: 3-30                 [2, 64, 14, 14]           30,912
    ├─Inception: 1-10                             [2, 512, 14, 14]          --
    │    └─BasicConv2d: 2-16                      [2, 160, 14, 14]          --
    │    │    └─Sequential: 3-31                  [2, 160, 14, 14]          82,400
    │    └─Sequential: 2-17                       [2, 224, 14, 14]          --
    │    │    └─BasicConv2d: 3-32                 [2, 112, 14, 14]          57,680
    │    │    └─BasicConv2d: 3-33                 [2, 224, 14, 14]          226,464
    │    └─Sequential: 2-18                       [2, 64, 14, 14]           --
    │    │    └─BasicConv2d: 3-34                 [2, 24, 14, 14]           12,360
    │    │    └─BasicConv2d: 3-35                 [2, 64, 14, 14]           38,592
    │    └─Sequential: 2-19                       [2, 64, 14, 14]           --
    │    │    └─MaxPool2d: 3-36                   [2, 512, 14, 14]          --
    │    │    └─BasicConv2d: 3-37                 [2, 64, 14, 14]           32,960
    ├─Inception: 1-11                             [2, 512, 14, 14]          --
    │    └─BasicConv2d: 2-20                      [2, 128, 14, 14]          --
    │    │    └─Sequential: 3-38                  [2, 128, 14, 14]          65,920
    │    └─Sequential: 2-21                       [2, 256, 14, 14]          --
    │    │    └─BasicConv2d: 3-39                 [2, 128, 14, 14]          65,920
    │    │    └─BasicConv2d: 3-40                 [2, 256, 14, 14]          295,680
    │    └─Sequential: 2-22                       [2, 64, 14, 14]           --
    │    │    └─BasicConv2d: 3-41                 [2, 24, 14, 14]           12,360
    │    │    └─BasicConv2d: 3-42                 [2, 64, 14, 14]           38,592
    │    └─Sequential: 2-23                       [2, 64, 14, 14]           --
    │    │    └─MaxPool2d: 3-43                   [2, 512, 14, 14]          --
    │    │    └─BasicConv2d: 3-44                 [2, 64, 14, 14]           32,960
    ├─Inception: 1-12                             [2, 528, 14, 14]          --
    │    └─BasicConv2d: 2-24                      [2, 112, 14, 14]          --
    │    │    └─Sequential: 3-45                  [2, 112, 14, 14]          57,680
    │    └─Sequential: 2-25                       [2, 288, 14, 14]          --
    │    │    └─BasicConv2d: 3-46                 [2, 144, 14, 14]          74,160
    │    │    └─BasicConv2d: 3-47                 [2, 288, 14, 14]          374,112
    │    └─Sequential: 2-26                       [2, 64, 14, 14]           --
    │    │    └─BasicConv2d: 3-48                 [2, 32, 14, 14]           16,480
    │    │    └─BasicConv2d: 3-49                 [2, 64, 14, 14]           51,392
    │    └─Sequential: 2-27                       [2, 64, 14, 14]           --
    │    │    └─MaxPool2d: 3-50                   [2, 512, 14, 14]          --
    │    │    └─BasicConv2d: 3-51                 [2, 64, 14, 14]           32,960
    ├─Inception: 1-13                             [2, 832, 14, 14]          --
    │    └─BasicConv2d: 2-28                      [2, 256, 14, 14]          --
    │    │    └─Sequential: 3-52                  [2, 256, 14, 14]          135,936
    │    └─Sequential: 2-29                       [2, 320, 14, 14]          --
    │    │    └─BasicConv2d: 3-53                 [2, 160, 14, 14]          84,960
    │    │    └─BasicConv2d: 3-54                 [2, 320, 14, 14]          461,760
    │    └─Sequential: 2-30                       [2, 128, 14, 14]          --
    │    │    └─BasicConv2d: 3-55                 [2, 32, 14, 14]           16,992
    │    │    └─BasicConv2d: 3-56                 [2, 128, 14, 14]          102,784
    │    └─Sequential: 2-31                       [2, 128, 14, 14]          --
    │    │    └─MaxPool2d: 3-57                   [2, 528, 14, 14]          --
    │    │    └─BasicConv2d: 3-58                 [2, 128, 14, 14]          67,968
    ├─MaxPool2d: 1-14                             [2, 832, 7, 7]            --
    ├─Inception: 1-15                             [2, 832, 7, 7]            --
    │    └─BasicConv2d: 2-32                      [2, 256, 7, 7]            --
    │    │    └─Sequential: 3-59                  [2, 256, 7, 7]            213,760
    │    └─Sequential: 2-33                       [2, 320, 7, 7]            --
    │    │    └─BasicConv2d: 3-60                 [2, 160, 7, 7]            133,600
    │    │    └─BasicConv2d: 3-61                 [2, 320, 7, 7]            461,760
    │    └─Sequential: 2-34                       [2, 128, 7, 7]            --
    │    │    └─BasicConv2d: 3-62                 [2, 32, 7, 7]             26,720
    │    │    └─BasicConv2d: 3-63                 [2, 128, 7, 7]            102,784
    │    └─Sequential: 2-35                       [2, 128, 7, 7]            --
    │    │    └─MaxPool2d: 3-64                   [2, 832, 7, 7]            --
    │    │    └─BasicConv2d: 3-65                 [2, 128, 7, 7]            106,880
    ├─Inception: 1-16                             [2, 1024, 7, 7]           --
    │    └─BasicConv2d: 2-36                      [2, 384, 7, 7]            --
    │    │    └─Sequential: 3-66                  [2, 384, 7, 7]            320,640
    │    └─Sequential: 2-37                       [2, 384, 7, 7]            --
    │    │    └─BasicConv2d: 3-67                 [2, 192, 7, 7]            160,320
    │    │    └─BasicConv2d: 3-68                 [2, 384, 7, 7]            664,704
    │    └─Sequential: 2-38                       [2, 128, 7, 7]            --
    │    │    └─BasicConv2d: 3-69                 [2, 48, 7, 7]             40,080
    │    │    └─BasicConv2d: 3-70                 [2, 128, 7, 7]            153,984
    │    └─Sequential: 2-39                       [2, 128, 7, 7]            --
    │    │    └─MaxPool2d: 3-71                   [2, 832, 7, 7]            --
    │    │    └─BasicConv2d: 3-72                 [2, 128, 7, 7]            106,880
    ├─AdaptiveAvgPool2d: 1-17                     [2, 1024, 1, 1]           --
    ├─Dropout: 1-18                               [2, 1024]                 --
    ├─Linear: 1-19                                [2, 1000]                 1,025,000
    ===============================================================================================
    Total params: 13,393,352
    Trainable params: 13,393,352
    Non-trainable params: 0
    Total mult-adds (G): 3.17
    ===============================================================================================
    Input size (MB): 1.20
    Forward/backward pass size (MB): 103.25
    Params size (MB): 28.05
    Estimated Total Size (MB): 132.51
    ===============================================================================================


<br/>


## **Forward Pass**


<br/>

```python
x = torch.randn(2,3,224,224)
aux1_out, aux2_out, out = model(x)
print(aux1_out.shape)
print(aux2_out.shape)
print(out.shape)
```

    torch.Size([2, 1000])
    torch.Size([2, 1000])
    torch.Size([2, 1000])

