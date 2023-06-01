---
layout: post
title : "[Paper Review & Implementation] Rethinking the Inception Architecture for Computer Vision (GoogleNet/Inception Net v2-3, 2015)"
img: papers/cnn/inception_v3.png
categories: [papers-cnn]  
tag : [Paper Review, CNN, GoogleNet, Inception Net v3, Implementation, PyTorch]
toc : true
toc_sticky : true
---


## **Outlines** 
- [**References**](#references)
- [**Inception v2-v3 Architecture**](#inception-v2-v3-architecture)
- [**Label Smoothing**](#label-smoothing)
- [**Implementation with PyTorch**](#implementation-with-pytorch)
- [**Model Summary**](#model-summary)
- [**Forward Pass**](#forward-pass)

<br/>


## **References**
- [Rethinking the Inception Architecture for Computer Vision, Christian Szegedy (2015)](https://arxiv.org/abs/1512.00567){:target="_blank"}
- [https://hyukppen.modoo.at/?link=5db82s6p](https://hyukppen.modoo.at/?link=5db82s6p){:target="_blank"}

<br/>

## **Inception v2-v3 Architecture**

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img width="400" alt="image" src="https://github.com/SuminizZ/Physics/assets/92680829/f64a93a6-9a39-448d-9eb6-b5976bbc7fc1"><br/>

**Figrue 5 :** <br/>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/8801af4a-48c3-4fb6-a67d-6d633359e160" width="250"><br/>
**Figrue 6:** <br/>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/0266e727-eca7-4005-a1d3-39dcebd1bbff" width="250"><br/>
**Figrue 7 :** <br/>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/60e7ccca-ce7a-4e3a-9d91-0b5b431f3da0" width="250"><br/>

<br/>

**1. Add alternative ways to factorize convolution and inception**

<img src="https://github.com/SuminizZ/Physics/assets/92680829/0083b71e-4e16-44f5-a483-2ae07d8d32ac" width="230px">
<img src="https://github.com/SuminizZ/Physics/assets/92680829/168e9439-36fc-4424-9e31-49d4ae75db19" width="230px">

- 5x5 -> 3x3 + 3x3 
    - avoid representational bottleneck (factorizing large receptive field into multiples of small ones)
    - parameter reduction : locally adjacent pixels share parameters
    - spatial aggregation : factorizing itself creates strong correlation between adjacent units (less loss of spatial information during reduction)

- nxn -> 1xn + nx1 
    - effect dramatically increases as n grows
    - works better for medium grid-sizes (feature maps)
    - 7 is suggested as optimal in the paper


**2. Parallel use of stride 2 convolution and pooling to decrease grid-size**

<img src="https://github.com/SuminizZ/Physics/assets/92680829/8d33a9ac-8ebc-4921-879f-803080aaea6b" width="400">

- Prevent representational bottlenecks and improve computational cost saving at the same time
- can reduce loss of representational information while decreasing the grid size

**3. Inception v2 -> v3 : Use RMSProp, Label Smoothing, BN-auxiliary**

- remove lower auxiliary layer (doesn't contribute to performance at all)
- but network with second (upper) aux classifier does overtake the network without aux near the end of the training (not in the early stages)
- but this effect doesn't seems to be related to increased gradient signals of lower features as expected in inception v1 paper.
- Instead, it seems to be associated with regularization effect due to the batch normalization added to side branch.
- BN acts as a regularizer by normalizing the activations (making sure subsequent layers to recieve inputs within a constant rage and prevent extreme activation, saturation, and inputs from entering into the non-linear regime of activation function) 

<br/>

## **Label Smoothing**

<img>

<br/>

## **Implementation with PyTorch**

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


class InceptionA(nn.Module):
    def __init__(self, in_channel, final_F):
        super().__init__()
        self.final_F = final_F
        self.branch_5x5 = nn.Sequential(BasicConv2d(in_channel, 64, 1, 1, padding=0),
                                        BasicConv2d(64, 64, 3, 1, padding=1),
                                        BasicConv2d(64, 96, 3, 1, padding=1))
        self.branch_3x3 = nn.Sequential(BasicConv2d(in_channel, 48, 1, 1, padding=0),
                                        BasicConv2d(48, 64, 3, 1, padding=1))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3,1,padding=1),
                                         BasicConv2d(in_channel, 64, 1, 1, padding=0))
        self.branch_1x1 = BasicConv2d(in_channel, 64, 1, 1, padding=0)

    def forward(self, x):
        x_concat = [self.branch_5x5(x), self.branch_3x3(x), self.branch_pool(x), self.branch_1x1(x)]
        x_concat = torch.cat(x_concat, 1)   # N,F,H,W
        assert x_concat.shape[1] == self.final_F
        
        return x_concat


class InceptionB(nn.Module):
    def __init__(self, in_channel, F, final_F):
        super().__init__()
        self.final_F = final_F
        self.branch_2n = nn.Sequential(BasicConv2d(in_channel, F, 1, 1, padding=0),
                                        BasicConv2d(F, F, (1,7), 1, padding=(0,3)),
                                        BasicConv2d(F, F, (7,1), 1, padding=(3,0)),
                                        BasicConv2d(F, F, (1,7), 1, padding=(0,3)),
                                        BasicConv2d(F, 192, (7,1), 1, padding=(3,0)))
        self.branch_n = nn.Sequential(BasicConv2d(in_channel, F, 1, 1, padding=0),
                                        BasicConv2d(F, F, (1,7), 1, padding=(0,3)),
                                        BasicConv2d(F, 192, (7,1), 1, padding=(3,0)))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3,1,padding=1),
                                         BasicConv2d(in_channel, 192, 1, 1, padding=0))
        self.branch_1x1 = BasicConv2d(in_channel, 192, 1, 1, padding=0)

    def forward(self, x):
        x_concat = [self.branch_2n(x), self.branch_n(x), self.branch_pool(x), self.branch_1x1(x)]
        x_concat = torch.cat(x_concat, 1)   # N,F,H,W
        assert x_concat.shape[1] == self.final_F
        
        return x_concat


class GridReduc(nn.Module):
    def __init__(self, in_channel, Fs, final_F):
        super().__init__()
        self.final_F = final_F
        F0, F1, F2 = Fs
        self.convstride_5x5 = nn.Sequential(BasicConv2d(in_channel, F0, 1, 1, padding=0),
                                            BasicConv2d(F0, F1, 3, 1, padding=1),
                                            BasicConv2d(F1, F1, 3, 2, padding=0))
        self.convstride_3x3 = nn.Sequential(BasicConv2d(in_channel, F0, 1, 1, padding=0),
                                            BasicConv2d(F0, F2, 3, 2, padding=0))
        self.poolstride = nn.MaxPool2d(3, 2, padding=0)

    def forward(self, x):
        x_concat = [self.convstride_5x5(x), self.convstride_3x3(x), self.poolstride(x)]
        x_concat = torch.cat(x_concat, 1)   # N,F,H,W
        assert x_concat.shape[1] == self.final_F
        
        return x_concat


class InceptionC(nn.Module):
    def __init__(self, in_channel, final_F):
        super().__init__()
        self.final_F = final_F
        self.branch_5x5 = nn.Sequential(BasicConv2d(in_channel, 448, 1, 1, padding=0),
                                        BasicConv2d(448, 384, 3, 1, padding=1))
        self.branch_5x5_1 = nn.Sequential(self.branch_5x5,
                                          BasicConv2d(384, 384, (1,3), 1, padding=(0,1)))
        self.branch_5x5_2 = nn.Sequential(self.branch_5x5,
                                          BasicConv2d(384, 384, (3,1), 1, padding=(1,0)))

        self.branch_3x3 = BasicConv2d(in_channel, 384, 1, 1, padding=0)
        self.branch_3x3_1 = nn.Sequential(self.branch_3x3,
                                          BasicConv2d(384, 384, (1,3), 1, padding=(0,1)))
        self.branch_3x3_2 = nn.Sequential(self.branch_3x3,
                                          BasicConv2d(384, 384, (3,1), 1, padding=(1,0)))

        self.branch_pool = nn.Sequential(nn.MaxPool2d(3,1,padding=1),
                                         BasicConv2d(in_channel, 192, 1, 1, padding=0))
        self.branch_1x1 = BasicConv2d(in_channel, 320, 1, 1, padding=0)

    def forward(self, x):
        branch_5x5_concat = torch.cat([self.branch_5x5_1(x), self.branch_5x5_2(x)], 1)
        branch_3x3_concat = torch.cat([self.branch_3x3_1(x), self.branch_3x3_2(x)], 1)

        x_concat = [branch_5x5_concat, branch_3x3_concat, self.branch_pool(x), self.branch_1x1(x)]
        x_concat = torch.cat(x_concat, 1)   # N,F,H,W
        assert x_concat.shape[1] == self.final_F
        
        return x_concat


class AuxOut(nn.Module):  
    def __init__(self, in_channel, p, num_classes):
        super().__init__()
        self.avgpool_conv = nn.Sequential(nn.AvgPool2d(5, 3, padding=0),
                                          BasicConv2d(in_channel, 1024, 1, 1, padding=0),
                                          nn.AdaptiveAvgPool2d((1,1)))
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.avgpool_conv(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out 


class Inception_V3(nn.Module):
    def __init__(self, init_weights=True, p=0.5, use_aux=True, in_channel=3, num_classes=1000):
        super().__init__()
        self.use_aux = use_aux

        self.conv1a = BasicConv2d(in_channel, 32, 3, 2, padding=0)
        self.conv1b = BasicConv2d(32, 32, 3, 1, padding=0)
        self.conv1c = BasicConv2d(32, 64, 3, 1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=0)

        self.conv2a = BasicConv2d(64, 80, 3, 1, padding=0)
        self.conv2b = BasicConv2d(80, 192, 3, 2, padding=0)
        self.conv2c = BasicConv2d(192, 288, 3, 1, padding=1)

        #  inceptionA x 2 + GridReduc x 1 : input = (35x35x288), output = (17x17x768)
        self.inceptionA_1 = InceptionA(288, 288)
        self.inceptionA_2 = InceptionA(288, 288)
        self.gridreducA = GridReduc(288, (64, 96, 384), 768)

        #  inceptionB x 4 + GridReduc x 1 : input = (17x17x768), output = (8x8x1280)
        self.inceptionB_1 = InceptionB(768, 128, 768)
        self.inceptionB_2 = InceptionB(768, 160, 768)
        self.inceptionB_3 = InceptionB(768, 160, 768)
        self.inceptionB_4 = InceptionB(768, 192, 768)
        self.gridreducB = GridReduc(768, (192, 192, 320), 1280)

        #  inceptionC x 2 : out feature maps 2048
        self.inceptionC_1 = InceptionC(1280, 2048)
        self.inceptionC_2 = InceptionC(2048, 2048)   # 8x8x2048

        if use_aux:
            self.aux = AuxOut(768, p, num_classes)
        else:
            self.aux = None
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=p)
        self.classifier = nn.Linear(2048, num_classes)

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
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv1c(x)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv2c(x)

        x = self.inceptionA_1(x)
        x = self.inceptionA_2(x)
        x = self.gridreducA(x)

        if self.training and self.aux is not None:
            aux_out = self.aux(x)
        else: aux_out = None

        x = self.inceptionB_1(x)
        x = self.inceptionB_2(x)
        x = self.inceptionB_3(x)
        x = self.inceptionB_4(x)
        x = self.gridreducB(x)

        x = self.inceptionC_1(x)
        x = self.inceptionC_2(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.classifier(x)

        return aux_out, out
```

<br/>

## **Model Summary**

<br/>

```python
model = Inception_V3()
summary(model, input_size=(2, 3, 299, 299), device='cpu')
```

<br/>


    ====================================================================================================
    Layer (type:depth-idx)                             Output Shape              Param #
    ====================================================================================================
    Inception_V3                                       --                        1,814,504
    ├─BasicConv2d: 1-1                                 [2, 32, 149, 149]         --
    │    └─Sequential: 2-1                             [2, 32, 149, 149]         --
    │    │    └─Conv2d: 3-1                            [2, 32, 149, 149]         896
    │    │    └─BatchNorm2d: 3-2                       [2, 32, 149, 149]         64
    │    │    └─ReLU: 3-3                              [2, 32, 149, 149]         --
    ├─BasicConv2d: 1-2                                 [2, 32, 147, 147]         --
    │    └─Sequential: 2-2                             [2, 32, 147, 147]         --
    │    │    └─Conv2d: 3-4                            [2, 32, 147, 147]         9,248
    │    │    └─BatchNorm2d: 3-5                       [2, 32, 147, 147]         64
    │    │    └─ReLU: 3-6                              [2, 32, 147, 147]         --
    ├─BasicConv2d: 1-3                                 [2, 64, 147, 147]         --
    │    └─Sequential: 2-3                             [2, 64, 147, 147]         --
    │    │    └─Conv2d: 3-7                            [2, 64, 147, 147]         18,496
    │    │    └─BatchNorm2d: 3-8                       [2, 64, 147, 147]         128
    │    │    └─ReLU: 3-9                              [2, 64, 147, 147]         --
    ├─MaxPool2d: 1-4                                   [2, 64, 73, 73]           --
    ├─BasicConv2d: 1-5                                 [2, 80, 71, 71]           --
    │    └─Sequential: 2-4                             [2, 80, 71, 71]           --
    │    │    └─Conv2d: 3-10                           [2, 80, 71, 71]           46,160
    │    │    └─BatchNorm2d: 3-11                      [2, 80, 71, 71]           160
    │    │    └─ReLU: 3-12                             [2, 80, 71, 71]           --
    ├─BasicConv2d: 1-6                                 [2, 192, 35, 35]          --
    │    └─Sequential: 2-5                             [2, 192, 35, 35]          --
    │    │    └─Conv2d: 3-13                           [2, 192, 35, 35]          138,432
    │    │    └─BatchNorm2d: 3-14                      [2, 192, 35, 35]          384
    │    │    └─ReLU: 3-15                             [2, 192, 35, 35]          --
    ├─BasicConv2d: 1-7                                 [2, 288, 35, 35]          --
    │    └─Sequential: 2-6                             [2, 288, 35, 35]          --
    │    │    └─Conv2d: 3-16                           [2, 288, 35, 35]          497,952
    │    │    └─BatchNorm2d: 3-17                      [2, 288, 35, 35]          576
    │    │    └─ReLU: 3-18                             [2, 288, 35, 35]          --
    ├─InceptionA: 1-8                                  [2, 288, 35, 35]          --
    │    └─Sequential: 2-7                             [2, 96, 35, 35]           --
    │    │    └─BasicConv2d: 3-19                      [2, 64, 35, 35]           18,624
    │    │    └─BasicConv2d: 3-20                      [2, 64, 35, 35]           37,056
    │    │    └─BasicConv2d: 3-21                      [2, 96, 35, 35]           55,584
    │    └─Sequential: 2-8                             [2, 64, 35, 35]           --
    │    │    └─BasicConv2d: 3-22                      [2, 48, 35, 35]           13,968
    │    │    └─BasicConv2d: 3-23                      [2, 64, 35, 35]           27,840
    │    └─Sequential: 2-9                             [2, 64, 35, 35]           --
    │    │    └─MaxPool2d: 3-24                        [2, 288, 35, 35]          --
    │    │    └─BasicConv2d: 3-25                      [2, 64, 35, 35]           18,624
    │    └─BasicConv2d: 2-10                           [2, 64, 35, 35]           --
    │    │    └─Sequential: 3-26                       [2, 64, 35, 35]           18,624
    ├─InceptionA: 1-9                                  [2, 288, 35, 35]          --
    │    └─Sequential: 2-11                            [2, 96, 35, 35]           --
    │    │    └─BasicConv2d: 3-27                      [2, 64, 35, 35]           18,624
    │    │    └─BasicConv2d: 3-28                      [2, 64, 35, 35]           37,056
    │    │    └─BasicConv2d: 3-29                      [2, 96, 35, 35]           55,584
    │    └─Sequential: 2-12                            [2, 64, 35, 35]           --
    │    │    └─BasicConv2d: 3-30                      [2, 48, 35, 35]           13,968
    │    │    └─BasicConv2d: 3-31                      [2, 64, 35, 35]           27,840
    │    └─Sequential: 2-13                            [2, 64, 35, 35]           --
    │    │    └─MaxPool2d: 3-32                        [2, 288, 35, 35]          --
    │    │    └─BasicConv2d: 3-33                      [2, 64, 35, 35]           18,624
    │    └─BasicConv2d: 2-14                           [2, 64, 35, 35]           --
    │    │    └─Sequential: 3-34                       [2, 64, 35, 35]           18,624
    ├─GridReduc: 1-10                                  [2, 768, 17, 17]          --
    │    └─Sequential: 2-15                            [2, 96, 17, 17]           --
    │    │    └─BasicConv2d: 3-35                      [2, 64, 35, 35]           18,624
    │    │    └─BasicConv2d: 3-36                      [2, 96, 35, 35]           55,584
    │    │    └─BasicConv2d: 3-37                      [2, 96, 17, 17]           83,232
    │    └─Sequential: 2-16                            [2, 384, 17, 17]          --
    │    │    └─BasicConv2d: 3-38                      [2, 64, 35, 35]           18,624
    │    │    └─BasicConv2d: 3-39                      [2, 384, 17, 17]          222,336
    │    └─MaxPool2d: 2-17                             [2, 288, 17, 17]          --
    ├─InceptionB: 1-11                                 [2, 768, 17, 17]          --
    │    └─Sequential: 2-18                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-40                      [2, 128, 17, 17]          98,688
    │    │    └─BasicConv2d: 3-41                      [2, 128, 17, 17]          115,072
    │    │    └─BasicConv2d: 3-42                      [2, 128, 17, 17]          115,072
    │    │    └─BasicConv2d: 3-43                      [2, 128, 17, 17]          115,072
    │    │    └─BasicConv2d: 3-44                      [2, 192, 17, 17]          172,608
    │    └─Sequential: 2-19                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-45                      [2, 128, 17, 17]          98,688
    │    │    └─BasicConv2d: 3-46                      [2, 128, 17, 17]          115,072
    │    │    └─BasicConv2d: 3-47                      [2, 192, 17, 17]          172,608
    │    └─Sequential: 2-20                            [2, 192, 17, 17]          --
    │    │    └─MaxPool2d: 3-48                        [2, 768, 17, 17]          --
    │    │    └─BasicConv2d: 3-49                      [2, 192, 17, 17]          148,032
    │    └─BasicConv2d: 2-21                           [2, 192, 17, 17]          --
    │    │    └─Sequential: 3-50                       [2, 192, 17, 17]          148,032
    ├─InceptionB: 1-12                                 [2, 768, 17, 17]          --
    │    └─Sequential: 2-22                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-51                      [2, 160, 17, 17]          123,360
    │    │    └─BasicConv2d: 3-52                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-53                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-54                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-55                      [2, 192, 17, 17]          215,616
    │    └─Sequential: 2-23                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-56                      [2, 160, 17, 17]          123,360
    │    │    └─BasicConv2d: 3-57                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-58                      [2, 192, 17, 17]          215,616
    │    └─Sequential: 2-24                            [2, 192, 17, 17]          --
    │    │    └─MaxPool2d: 3-59                        [2, 768, 17, 17]          --
    │    │    └─BasicConv2d: 3-60                      [2, 192, 17, 17]          148,032
    │    └─BasicConv2d: 2-25                           [2, 192, 17, 17]          --
    │    │    └─Sequential: 3-61                       [2, 192, 17, 17]          148,032
    ├─InceptionB: 1-13                                 [2, 768, 17, 17]          --
    │    └─Sequential: 2-26                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-62                      [2, 160, 17, 17]          123,360
    │    │    └─BasicConv2d: 3-63                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-64                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-65                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-66                      [2, 192, 17, 17]          215,616
    │    └─Sequential: 2-27                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-67                      [2, 160, 17, 17]          123,360
    │    │    └─BasicConv2d: 3-68                      [2, 160, 17, 17]          179,680
    │    │    └─BasicConv2d: 3-69                      [2, 192, 17, 17]          215,616
    │    └─Sequential: 2-28                            [2, 192, 17, 17]          --
    │    │    └─MaxPool2d: 3-70                        [2, 768, 17, 17]          --
    │    │    └─BasicConv2d: 3-71                      [2, 192, 17, 17]          148,032
    │    └─BasicConv2d: 2-29                           [2, 192, 17, 17]          --
    │    │    └─Sequential: 3-72                       [2, 192, 17, 17]          148,032
    ├─InceptionB: 1-14                                 [2, 768, 17, 17]          --
    │    └─Sequential: 2-30                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-73                      [2, 192, 17, 17]          148,032
    │    │    └─BasicConv2d: 3-74                      [2, 192, 17, 17]          258,624
    │    │    └─BasicConv2d: 3-75                      [2, 192, 17, 17]          258,624
    │    │    └─BasicConv2d: 3-76                      [2, 192, 17, 17]          258,624
    │    │    └─BasicConv2d: 3-77                      [2, 192, 17, 17]          258,624
    │    └─Sequential: 2-31                            [2, 192, 17, 17]          --
    │    │    └─BasicConv2d: 3-78                      [2, 192, 17, 17]          148,032
    │    │    └─BasicConv2d: 3-79                      [2, 192, 17, 17]          258,624
    │    │    └─BasicConv2d: 3-80                      [2, 192, 17, 17]          258,624
    │    └─Sequential: 2-32                            [2, 192, 17, 17]          --
    │    │    └─MaxPool2d: 3-81                        [2, 768, 17, 17]          --
    │    │    └─BasicConv2d: 3-82                      [2, 192, 17, 17]          148,032
    │    └─BasicConv2d: 2-33                           [2, 192, 17, 17]          --
    │    │    └─Sequential: 3-83                       [2, 192, 17, 17]          148,032
    ├─GridReduc: 1-15                                  [2, 1280, 8, 8]           --
    │    └─Sequential: 2-34                            [2, 192, 8, 8]            --
    │    │    └─BasicConv2d: 3-84                      [2, 192, 17, 17]          148,032
    │    │    └─BasicConv2d: 3-85                      [2, 192, 17, 17]          332,352
    │    │    └─BasicConv2d: 3-86                      [2, 192, 8, 8]            332,352
    │    └─Sequential: 2-35                            [2, 320, 8, 8]            --
    │    │    └─BasicConv2d: 3-87                      [2, 192, 17, 17]          148,032
    │    │    └─BasicConv2d: 3-88                      [2, 320, 8, 8]            553,920
    │    └─MaxPool2d: 2-36                             [2, 768, 8, 8]            --
    ├─InceptionC: 1-16                                 [2, 2048, 8, 8]           --
    │    └─Sequential: 2-37                            [2, 384, 8, 8]            2,567,744
    │    └─Sequential: 2-38                            --                        (recursive)
    │    │    └─Sequential: 3-89                       [2, 384, 8, 8]            2,124,224
    │    └─Sequential: 2-39                            --                        (recursive)
    │    │    └─BasicConv2d: 3-90                      [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-40                            [2, 384, 8, 8]            2,124,224
    │    │    └─Sequential: 3-91                       [2, 384, 8, 8]            (recursive)
    │    │    └─BasicConv2d: 3-92                      [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-41                            [2, 384, 8, 8]            936,192
    │    └─Sequential: 2-42                            --                        (recursive)
    │    │    └─BasicConv2d: 3-93                      [2, 384, 8, 8]            492,672
    │    └─Sequential: 2-43                            --                        (recursive)
    │    │    └─BasicConv2d: 3-94                      [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-44                            [2, 384, 8, 8]            492,672
    │    │    └─BasicConv2d: 3-95                      [2, 384, 8, 8]            (recursive)
    │    │    └─BasicConv2d: 3-96                      [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-45                            [2, 192, 8, 8]            --
    │    │    └─MaxPool2d: 3-97                        [2, 1280, 8, 8]           --
    │    │    └─BasicConv2d: 3-98                      [2, 192, 8, 8]            246,336
    │    └─BasicConv2d: 2-46                           [2, 320, 8, 8]            --
    │    │    └─Sequential: 3-99                       [2, 320, 8, 8]            410,560
    ├─InceptionC: 1-17                                 [2, 2048, 8, 8]           --
    │    └─Sequential: 2-47                            [2, 384, 8, 8]            2,911,808
    │    └─Sequential: 2-48                            --                        (recursive)
    │    │    └─Sequential: 3-100                      [2, 384, 8, 8]            2,468,288
    │    └─Sequential: 2-49                            --                        (recursive)
    │    │    └─BasicConv2d: 3-101                     [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-50                            [2, 384, 8, 8]            2,468,288
    │    │    └─Sequential: 3-102                      [2, 384, 8, 8]            (recursive)
    │    │    └─BasicConv2d: 3-103                     [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-51                            [2, 384, 8, 8]            1,231,104
    │    └─Sequential: 2-52                            --                        (recursive)
    │    │    └─BasicConv2d: 3-104                     [2, 384, 8, 8]            787,584
    │    └─Sequential: 2-53                            --                        (recursive)
    │    │    └─BasicConv2d: 3-105                     [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-54                            [2, 384, 8, 8]            787,584
    │    │    └─BasicConv2d: 3-106                     [2, 384, 8, 8]            (recursive)
    │    │    └─BasicConv2d: 3-107                     [2, 384, 8, 8]            443,520
    │    └─Sequential: 2-55                            [2, 192, 8, 8]            --
    │    │    └─MaxPool2d: 3-108                       [2, 2048, 8, 8]           --
    │    │    └─BasicConv2d: 3-109                     [2, 192, 8, 8]            393,792
    │    └─BasicConv2d: 2-56                           [2, 320, 8, 8]            --
    │    │    └─Sequential: 3-110                      [2, 320, 8, 8]            656,320
    ├─AdaptiveAvgPool2d: 1-18                          [2, 2048, 1, 1]           --
    ├─Dropout: 1-19                                    [2, 2048]                 --
    ├─Linear: 1-20                                     [2, 1000]                 2,049,000
    ====================================================================================================
    Total params: 38,345,632
    Trainable params: 38,345,632
    Non-trainable params: 0
    Total mult-adds (G): 11.19
    ====================================================================================================
    Input size (MB): 2.15
    Forward/backward pass size (MB): 259.10
    Params size (MB): 92.05
    Estimated Total Size (MB): 353.29
    ====================================================================================================


<br/>


## **Forward Pass**


<br/>

```python
x = torch.randn(2,3,299,299)
aux_out, out = model(x)
print(aux_out.shape)
print(out.shape)
```

<br/>

    torch.Size([2, 1000])
    torch.Size([2, 1000])

