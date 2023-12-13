---
layout: post
title : "[Paper Review] Visualizing the Loss Landscape of Neural Nets (Loss Landscape, 2018)"
img: papers/opt/loss_landscape.png
categories: [papers-opt]  
tag : [Paper Review, Loss Landscape, Optimization, Explainable AI]
toc : true
toc_sticky : true
---

## **Outlines** 
<br/>

- [**References**](#references)
- [**Varying Trainability of Networks Architectures**](#varying-trainability-of-networks-architectures)
- [**Basic Visualizations of Loss Function**](#basic-visualizations-of-loss-function)
- [**Filter-Wise Normalizaton**](#filter-wise-normalizaton)
- [**Understanding the Effect of Network Architecture on Geometry of Loss Landscapes**](#understanding-the-effect-of-network-architecture-on-geometry-of-loss-landscapes)
- [**Measurements of the Level of Convexity : Principle Curvatures**](#measurements-of-the-level-of-convexity---principle-curvatures)
- [**Visualization of Optimization Paths Using PCA Directions**](#visualization-of-optimization-paths-using-pca-directions)

<br/>

## **References**

- [Visualizing the Loss Landscape of Neural Nets, Hao Li1 (2018)](https://arxiv.org/pdf/1712.09913.pdf){:target="_blank"}
- [Qualitatively characterizing neural network optimization problems, Ian J. Goodfellow (2015)](https://arxiv.org/abs/1412.6544){:target='_blank'}

<br/>

## **Varying Trainability of Networks Architectures**
<br/>

- ResNet successfully address the degradation issue of deeper layers where training performance tends to decay with the depth of neural networks by introducing a novel architecture design named "skip-connection". Authors of the paper explained the reason behind the poorer trainability of deeper networks than its shallower counterpart is that deeper networks have difficulties in approximating identity mappings and to deal with this, they added some shortcut paths that directly connects the input to output of 2 or more layers that only fits the residual part (gap between the input and desired underlying output). Letting the networks to fit complicated functions only for residuals and simply adding the input to that residual mappings improve the training accuracy of the networks even with very deep structure over than 100 layers. This examples tells the trainability of networks is highly dependent of the architecture design choices. However, the fundamental mechanism of how they affects the performance of networks has not been clearly explained.  

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

&emsp;&emsp;- 1D linear interpolation of $\large f(\alpha) = L(\theta_s + \alpha(\theta_l - \theta_s))$ obtained by small-batch ($\large \theta^{s}$, batch_size = 128) and large-batch ($\large \theta^{l}$, batch_size = 8192). Top row (a, b, c) shows results without weight decay and bottom (d, e, f) with non-zero weight decay. 

&emsp;&emsp;- It is well defined that training networks with smaller batch size tends to have flat and wide solution due to a regularizing effect of randomness from small sample size and larger batch size, on the other hand, gives relatively sharper solution, which can be shown at the top row (with WD) of the Figure 2. 

&emsp;&emsp;- But this sharpness balance between $\large \theta^{s}$ and $\large \theta^{l}$ is flipped with non-zero weight decay (WD = 5e-4). Seeing Figure 2.-(d), loss plot with small batch size (6%) seems to have sharper and narrower shape compared to large batch size (10%). Besides, small-batch methods produce more large weights with zero weight decay and more small weights with non-zero weight decay (Figrue 2. b, c, e, f), which is obvious considering that networks with small batch size update the weight more frequently per epoch, thus more affected by weight decay. These results verify that sharpness is not fixed and can be flipped simply by turning on weight decay. 

&emsp;&emsp;- One can expect that as the sharpness is flipped, generalization performance is also flipped between  $\large \theta^{s}$ and $\large \theta^{l}$, but this is not the case and the small batch size still gives better generalization regardless of the curvature of loss plot.

&emsp;&emsp;- These results tells that sharpness of loss curvature drawn without filter normalizationi has no apparent correlation with generalization and sharpness comparison doesn't give meaningful interpretation about the model performance.

 
<br/>


## **Filter-Wise Normalizaton**

<br/>

- This paper mainly uses the second visualization approach described above but with a novel method filter normalizaton that successfully address the limitations of choosing random direction vectors $\large \delta$ and $\large \eta$.

- This limitation is due to the **scale invariance** of network weights where the network's performance remains unaffected when the weights of the network are scaled by a constant factor.

- This applies to the networks using **ReLU activation** that is not affected by the scale of inputs. What only matters is whether the input is greater or smaller than zero. 

- Scale invariance becomes more pronounced especially when adapting a **batch normalization** that re-scales the output of each layer (in case of pre-activation, input) and then pass it to the ReLU non-linearities. 

- This invariance is valid except some artifacts caused by overall magnitude of weights, which means large weights tend to be resilient to small perturbation while smaller weights shows more senitive responses to the same amount of perturbation. But this kind of variation is simply an artifact, not induced by intrinsic geometry of loss function, which what we are interested in.

- **To remove this scailing effect**, the paper uses filter-wise normalized direction vectors.
  - Firstly, produces two direction vector from random gaussian distribution with dimensions compatible with $\large \theta$.

  - Then, re-scales the weight of each filter ($\large d_{i,j}$) to have the same norm of the corresponding filter in $\large \theta_{i,j}$.

      &emsp;&emsp; $\large d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|} \cdot \|\theta_{i,j}\|$

  - Think of FC layer as equivalent to a Conv layer with 1x1 receptive fields and filter corresponds to the weigth matrix that generates one node (neuron).

  - Then draw the same 2D contour plots using filter-normalized direction vectors.

&emsp;&emsp;**Figure 3.**

&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/c4ebe556-83ea-4367-b063-86fcac202059" width="780">

  - They draw both 1D and 2D visualization both applied with filter normalization and the difference of sharpness between small and large batch size seems to be more subtle than before without filter normalization (Figure 2.). In summary, small batch size with non-zero weight decay (c) has wide and flat contours and lowest generalization error while larger batch size without weight decay (b) has sharp and narrow countours and highest generalization error. Now, sharpness correlates well with generalization error and side-by-side comparison of minimizers using visualized loss landscape become reasonalbe. 
  

<br/>

## **Understanding the Effect of Network Architecture on Geometry of Loss Landscapes**

<br/>

#### **1. Skip Connection and Depth**

<br/>

&emsp;&emsp;**Figure 5. : The effect of Skip Connection on Non-Convexity**

&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/aa8341fa-6d58-4141-b77e-074718907960" width="750">

<br/>

- Comparision between loss surface of ResNet (skip connection) and VGG-like (no skip connection) nets with incresing network depth from 20 to 110 layers optimized for CIFAR-10.

- Increasing the depth of ResNet network with skip-connection doesn't necessarily result in corresponding increase of convexities but the convexities of networks without skip connection (VGG-like nets) dramatically grows with the networks depth. 

- This transition from nearly convex to chaotic in response to increasing depth is way more prominent in VGG-like nets compared to ResNet with skip connection.

- In chaotic loss surface, there's not much region where gradient direction points towards the global optima at the center. This partitioning of chaotic and convex regions can explain the importance of weight initialization as it sets the starting point of converging trajectory. Well-behaved loss landscapes dominated by large, flat, and nearly convex regions has weaker dependence on weight initialization because most gradients are directed to the central minimizer regardless of its starting point. 
 
- Considering that introducing skip connection to deeper networks mitigates the pathological non-convexities of loss surface and helps weights converge faster towards the minimizer, architecture design choices of neural network has great impact on trainability of networks. 


<br/>

#### **2. Width of Networks : The Number of Filters Per Layer**

<br/>

&emsp;&emsp;**Figure 6. : Wide-ResNet-56 on CIFAR-10 with (top row) and without skip-connection (bottom row)**

&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/229ad560-37f1-45eb-a0ba-5a03e6c69ab3" width="750">

- Increasing the width of layer with bigger k (multipliers of the number of filters) results in more convex geometry of loss landscapes both for with and without skip-connection.


<br/>

## **Measurements of the Level of Convexity - Principle Curvatures**

<br/>

- Evaluation of convexity only by visualized geometry of loss landscape can be not so accurate as the visualization is achieved with extreme dimensionality reduction. Hence, the paper proposes other method to capture hidden non-convexity that low dimensional visualization missed.

- Negative most eigenvalues of hessian matrix (second derivative of loss with respect to parameters ($\large \alpha$, $\large \beta$)) can be used to determine the level of convexity. (truly convex function has positive or positive semi-definite hessian matrix) 

- However, covexity verified with dimensionality reduced plot can't guarantee the convexity of the loss plot of original dimensionality while vice versa can be possible. Rather, presence of convexity in low dimensional plot means the dominance of postiveness in high dimensional plot. 

- To capture the significance of convexity, the paper uses the ratio of minimum eigenvalue to maximum eigenvalue of hessian. 
    
    &emsp;&emsp;$\large \|\frac{\lambda_{\text{min}}}{\lambda_{\text{max}}}\|$

    <img src="https://github.com/SuminizZ/Physics/assets/92680829/9c462829-de85-4576-98d1-f38a749b76be" width="800">

    - Blue regions represents insignificant non-convexities compared to convexities, while yellow indicates significant level of negative curvature. (same minimizers and same direction vectors are used) 

    - As expected, seemingly convex and chaotic regions in the visualized loss surface do correspond to the blue colored regions and yellow colored regions, respectively. 


<br/>

## **Visualization of Optimization Paths Using PCA Directions**

<br/>

- The problem of using randomly selected direction vectors is that those vectors are highly likely to be orthogonal to optimization trajectory.

- This is because two random vectors are in extremely high dimensional space (the nubmer of parameters used can range from a few miliions or even bilions), while the trajectory lie in extremely low dimensional space (1 or 2). High dimensional vectors are orthogonal to random vectors with very high probability (called "blessing of dimensionality") and thus, orthogonal to low dimensional optimization trajectories.
   
&emsp;&emsp;&emsp;&emsp;**Figure 8. : Visualization of 2 dimensional optimizer trajectories using high-dimensional random direction vectors**

&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/79891a8c-0cb1-4cf5-9e97-2dfa5843f076" width="600">

<br/>

- To visualize the trajectory onto the loss surface, one can perform PCA to find a linear subspace that maximally preserves the original shape of optimization trajectory. 
    - First, creates the matrix $\large M = [\theta_0 - \theta_n; \dots ; \theta_{n-1} - \theta_n]$ and apply PCA to find two most explanatory components (2 eigenvectors with first and second biggest eigenvalues)
    
    - Then plot loss surface and optimization trajectories along these choosed directions.

    &emsp;&emsp;**Figure 9. : Visualization of Optimization Paths using PCA Directions**

    &emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/68f8d3ed-2680-43e8-921b-f8a7825f2a5d" width="780">

    - Now you can see newly constructed directions using PCA can capture roughly 50% ~ 90% of original shape of optimization paths. 


------------------

<br/>

&emsp;&emsp;To summarize, this paper tries to answer why are some architectures are easier to train than others by showing how neural architecture affects the geometry of loss landscapes with a novel visualization method called filter normalization. 
