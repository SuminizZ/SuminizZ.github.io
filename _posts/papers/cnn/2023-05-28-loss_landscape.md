---
layout: post
title : "[Paper Review] Visualizing the Loss Landscape of Neural Nets (Loss Landscape, 2018)"
img: papers/xai/loss_landscape.png
categories: [papers-xai]  
tag : [Paper Review, Loss Landscape, Interpretability & Explainability]
toc : true
toc_sticky : true
---

## **Outlines** 
- [**References**](#references)

<br/>

## **References**
- [ Visualizing the Loss Landscape of Neural Nets, Hao Li1 (2018)](https://arxiv.org/pdf/1712.09913.pdf){:target="_blank"}

<br/>

## **Varying Trainability of Networks Architectures**

- ResNet successfully address the degradation issue of deeper layers where training performance tends to decay with the depth of neural networks by introducing a novel architecture design named "skip-connection". Authors of the paper explained the reason behind the poorer trainability of deeper networks than its shallower counterpart is that deeper networks have difficulties in approximating identity mappings and to deal with this, they added some shortcut paths that directly connects the input to output of 2 or more layers that only fits the residual part (gap between the input and desired underlying output). Letting the networks to fit complicated functions only for residuals and simply adding the input to that residual mapping improves the training accuracy of the networks even with very deep structure. The fundamental mechanism of this improvement, however, has not been clearly explained.  


  dramatically improves the trainability of ne