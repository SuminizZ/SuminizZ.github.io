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
<br/>

- [**References**](#references)

<br/>

## **References**

<br/>

- [Visualizing the Loss Landscape of Neural Nets, Hao Li1 (2018)](https://arxiv.org/pdf/1712.09913.pdf){:target="_blank"}
- [Qualitatively characterizing neural network optimization problems, Ian J. Goodfellow (2015)](https://arxiv.org/abs/1412.6544){:target='_blank'}

<br/>

## **Varying Trainability of Networks Architectures**
<br/>

- ResNet successfully address the degradation issue of deeper layers where training performance tends to decay with the depth of neural networks by introducing a novel architecture design named "skip-connection". Authors of the paper explained the reason behind the poorer trainability of deeper networks than its shallower counterpart is that deeper networks have difficulties in approximating identity mappings and to deal with this, they added some shortcut paths that directly connects the input to output of 2 or more layers that only fits the residual part (gap between the input and desired underlying output). Letting the networks to fit complicated functions only for residuals and simply adding the input to that residual mappings improve the training accuracy of the networks even with very deep structure over than 100 layers. This examples tells the trainability of networks is highly dependent of the architecture design choices. However, the fundamental mechanism of they affects the performance of networks has not been clearly explained.  

- This paper provides a variety of visualizations for the loss landscapes of multiple networks architectures (e.g. VGG, ResNet, WideNet), helping intuitive understanding of how the geometry of neural loss function affects the generalization error and trainabiltiy of the networks.

- They introduce a simple method "Filter Normalization" that helps proving a strong correlation between the curvature (sharpness) of loss function and generalization errors of the networks, enabling side-by-side comparison of the minimizers of different networks architectures.

<br/>

&emsp;&emsp;&emsp;<img width="600" alt="image" src="https://github.com/SuminizZ/Physics/assets/92680829/141f483e-cb1e-48c0-b7ff-2ccd2c525870">

<br/>

## **Basic Visualizations of Loss Function**

<br/>

### &emsp;**1. One-Dimensional Linear Interpolation**
<br/>
  
  - strategy taken by Goodfellow in 2015 [(https://arxiv.org/abs/1412.6544)](https://arxiv.org/abs/1412.6544){:target='_blank'}
  - choose two sets of parameters $\large \theta$ and $\large \theta^{\prime}$, and simply evaluate the loss ($\large J(\theta)$) at a series of points along the line $\large e^{\theta(\alpha)} = (1-\alpha)\theta + \alpha\theta^{\prime}$ for varing $\alpha$, which is a scailing parameter.

&emsp;&emsp;&emsp;<img width="470" alt="image" src="https://github.com/SuminizZ/Physics/assets/92680829/c60840ec-620a-4746-8410-be2217df0d1b">


  - Even though this approach provides relatively simple and general visualization of how sharp or flattened the loss function is, it is very difficult to express non-covexities using 1D plots. 

<br/>

### &emsp;**2. 2D-Contour Plots & Random Directions**

- Plots the loss with respect to $\large \alpha$ and $\large \beta$, which both are scailing factors for two random direction vectors $\large \delta$ and $\large \eta$, respectively.
- Need to choose a center point $\large \theta^{*}$ in the graph, typically a minimizer of the networks

&emsp;&emsp;&emsp;&emsp;$\large f(\alpha, \beta) = L(\theta^* + \alpha\delta + \beta\eta)$

- Here, choose $\large \delta$ and $\large \eta$ from random gaussian distribution but with same dimension with parameters space.

- This approach enalbes 2-dimensioanl visualization of complex non-convexities of loss landscape. But still it has clear weakness in that selecting random direction vectors fails to capture a meaningful correlation between the geometry of loss surfaces and the generalization properties of the network. 

<br/>

### &emsp;**No Apparent Correlation between Sharpness and Generalization**
<br/>

&emsp;&emsp;&emsp;**Figure 2.** 

&emsp;&emsp;<img width="780" alt="image" src="https://github.com/SuminizZ/Physics/assets/92680829/6cd75cef-820c-4178-b947-d4d5c95a3a90">

<br/>

&emsp;&emsp;- 1D linear interpolation of $\large f(\alpha) = L(\theta_s + \alpha(\theta_l - \theta_s))$ obtained by small-batch ($\large \theta^{s}$, batch_size = 128) and large-batch ($\large \theta^{l}$, batch_size = 8192)

&emsp;&emsp;- It is well defined that training networks with smaller batch size tends to have flat and wide solution due to a regularizing effect of randomness from small sample size and larger batch size, on the other hand, gives relatively sharper solution, which can be shown at the top of the Figure 2. 



<br/>


## **Filter-Wise Normalizaton**

<br/>

- This paper mainly uses the second visualization approach described above but with a novel method filter normalizaton that successfully address the limitations of choosing random direction vectors $\large \delta$ and $\large \eta$.

- This limitation is due to the **scale invariance** of network weights where the network's performance remains unaffected when the weights of the network are scaled by a constant factor.

- This applies to the networks using **ReLU activation** that is not affected by the scale of inputs. What only matters is whether the input is greater or smaller than zero. 

- Scale invariance becomes more pronounced especially when adapting a **batch normalization** that re-scales the output of each layer (in case of pre-activation, input) and then pass it to the ReLU non-linearities. 

- This invariance is valid with some artifacts caused by overall magnitude of weights, which means large weights tend to be resilient to small perturbation while smaller weights shows more senitive responses to the same amount of perturbation. But this kind of variation is simply an artifact, not induced by intrinsic geometry of loss function, which what we are interested in.

- **To remove this scailing effect**, the paper uses filter-wise normalized direction vectors.
  - Firstly, produces two direction vector from random gaussian distribution with dimensions compatible with $\large \theta$.

  - Then, re-scales the weight of each filter ($\large d_{i,j}$) to have the same norm of the corresponding filter in $\large \theta_{i,j}$.

      &emsp;&emsp; $\large d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|} \cdot \|\theta_{i,j}\|$

  - Think of FC layer as equivalent to a Conv layer with 1x1 receptive fields and filter corresponds to the weigth matrix that generates one node (neuron).

  - Then draw the same 2D contour plots using filter-normalized direction vectors.

