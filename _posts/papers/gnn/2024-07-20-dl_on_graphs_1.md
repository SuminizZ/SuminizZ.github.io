---
layout: post
title : "[Paper Review] Deep Learning on Graphs: A Survey (Review, 2020) - Part 1. Graph RNNs"
img: papers/gnn/dlongraph.png
categories: [papers-gnn]  
tag : [Paper Review, Graph, GNN]
toc : true2020
toc_sticky : true
---

## **Outlines**
- [**Reference**](#reference)
- [**Deep Learning for Graph Data**](#deep-learning-utilized-to-analyze-graph-data)
- [**0. Notations and Preliminaries**](#0-notations-and-preliminaries)
- [**1. Graph Recurrent Neural Networks (RNNs)**](#1-graph-recurrent-neural-networks-rnns)
- [**Graph Convolutional Networks (CNNs)**](#graph-convolutional-networks-cnns)
- [**Graph AutoEncoders (GAE)**](#graph-autoencoders-gaes)
- [**Graph Reinforcement Learning (RL)**](#graph-reinforcement-learning-rl)
- [**Graph Adversarial Methods**](#graph-adversarial-methods)

<br/>


## **Reference**

<br/>

- [**Deep Learning on Graphs: A Survey, Zhang et al, 2020**](https://arxiv.org/pdf/1812.04202.pdf){:target="_blank"}
- [**The Graph Neural Network Model, Scarselli et al, 2009**](https://ro.uow.edu.au/cgi/viewcontent.cgi?article=10501&context=infopapers){:target="_blank"}

<br/>

## **Deep Learning for Graph Data**

<br/>

- Graphs are ubiquitous form of data in real world, including social networks, e-commerce networks, biology networks, traffic networks, and so on. 

    <img width="600" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/4dcc6914-64b8-411d-b648-7ed536955b21">
    
    <br/>

- Despite the prevalence of graph data, traditional deep learning approaches are not easily applicable to graphs since they are primarily designed for structured and sequential data.

- Here are some of the key challenges for adopting deep learning on graph.

    - **Geometric deep learning problem** : Hard to adjust existing deep learning to irregular and dynamic graph structures.
    
    - **Heterogenity and diversity of grpahs** : heterogenous/homogenous (types of nodes and edges), singed/unsigned (presence of signs associated with edges), static/dynamic (changes by time)
    
    - **Cost-Intensive Graph Representation** : Single graph is typically representd using muliple matrices, including adjacency matrix, node feature matrix, laplacian matrix, and so on. Carrying all these matrices can cause severe memory traffic and computational overhead. 
    
    - **Scalability** : Efficient algorithms and techniques are required to handle large-scale graph data.
    
    - **Graph Labeling and Supervised Learning** : Unlike image data or text data where each data point has clear label, assigning labels to nodes or edges is too ambiguous and requires domain expertise, which is also time and cost consuming. 

<br/>

#### **Five Categories of Existing Deep Learning Methods on Graphs**

<br/>

- Substantial research efforts have been made to tackle these challenges and apply deep learning methods to graphs. 

- Specifically, the methods can be classified as 5 categories, **Graph RNNs**, **GCNs**, **GAEs**, **Graph RL**, and **graph adversarial methods**. 

&emsp; <img width="950" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/03369d64-73ff-4583-957d-36bfef662b9c">


- This paper provides systematic and detailed overview of the prior works and connections between them.

- Note that this paper mainly focus on unsigned graphs.

<br/>


## **0. Notations and Preliminaries**

<br/>

### **0.1. Notations**

<img width="500" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/b1438f1d-9090-435c-b681-34a8a1ea6b9c">

- **V, E** : nodes and edges

- **Laplacian Matrix**

    &emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/02ee8961-a60d-40f9-b9ca-198fedde1e88" width="480">

    - $\large D - A$

        - $\large D(i, i) = \sum_{j} A(i, j)$

        - A : Adjacency matrix whose element (i, j) is 1 if node i, j is connected and 0 if not.

    - Eigendecomposition of laplacian matrix is denoted as $\large L\,=\,Q\Lambda Q^{T}$ where $\large \Lambda$ is a diagonal matrix containing eigenvalues and $\large Q$ is corresponding eigenvectors.

- **Transition Matrix**

    - $\large P(i, j)$ represents the probability of random walk from node $\large v_{i}$ to $\large v_{j}$. 


- $\large \mathcal{N}_{k}(i)$ 

    - K-step neighbors of node $\large v_{i}$ where  $\large \mathcal{D}(i, j)$, the shortest distance from node  $\large v_{i}$ to  $\large v_{j}$, is smaller than  $\large k$

    - <img width="220" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/91d785ef-daa3-40ed-9b2e-cfb85c26f57a">

- $\large \rho(\.)$ 

    - A general element-wise non-linear activation such as sigmoid or ReLU.

<br/>

### **0.2. Tasks for Deep Learning on Graph**

<br/>

- **Node-Focused tasks**

    - Associated with individual nodes in the graph. Examples include node classification, link prediction, and node recommendation.

- **Graph-Focused tasks** 

    -  Associated with the entire graph. Examples include graph classification, estimating various graph properties, and generating graphs.

 
<br/>

## **1. Graph Recurrent Neural Networks (RNNs)**

<br/>
   
- RNNs refer to recurrent networks such as gated recurrent networks (GRU) and LSTM mainly designed for processing sequential data. 

- Graph RNNs aim to capture recursive and sequential patterns of graphs. 

- Broadly divided into 2 categories, node-level RNNs (node-level pattter modeled by node states) at and graph level RNNs (graph-level pattern residing in global graph state).

&emsp;<img width="1000" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/90ec9371-98d8-4387-ab74-cbce6fca8eda">
 
<br/>

### **1.1. Node-Level Graph RNNs**

<br/>

#### **1.1.1. Graph Neural Networks (GNNs)**

<br/>

- Extends a existing recurrent neural network for processing the data represented in graph domains.

- Encode graph structural information where the node $\large v_{n}$ is represented as a low-dimensional state vector $\large x_{n}$ (denoted as $\large s_{n}$ in paper)
   
&emsp; <img width="470" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/d803a89b-6d91-4357-a8d6-53d8ceca42bb">


<br/>

##### **1.1.1.1. Positional Graphs**

<br/>

- Positional graphs incorporate explicit positional information or spatial coordinates associated with each node in the graph.

- Example : Molecular structures where each node represent a specific atom.

- How to compute $\large x_{n}(t+1)$ : 

    <img width="350" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/a92110fd-6dde-4470-8a5a-69172211ae02">

    <img width="700" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/53917e4a-58a8-4e51-87cf-f630d61f626d">

    <img width="600" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/173e95ee-91c9-46ac-88a7-593a513c67e1">


- Both $\large f_{w}$ and $\large g_{w}$ are learned feed forward networks.

- $\large f_{w}$ : parametric function to be learned called as local transition function.

    - Expresses the dependence of a node on its neighborhood. 

    - Assumed to be a **contraction map** with respect to the state as a sufficient condition for the existence and uniqueness of the solution of a system.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/6a8ac1c0-17da-4a40-b452-5386fb482272" width="750">    

- $\large g_{w}$ : Local output function that describes how the output is produced.

    - Computed locally at each node based on the unit state.

- $\large l_{n}, l_{co[n]}, x_{ne[n]}, l_{ne[n]}$ : label of node n, label of its edges, states and the labels of its neighborhoods, respectively. 

    - stacking up all the labels and states of the node and its neighborhoods. 

    - Sorted according to neighbors’ positions and is properly padded with special null values in positions corresponding to nonexisting neighbors

<br/>

##### **1.1.1.2. Non-Positional Graphs**

<br/>

- Non-Positional graph doesn't include explicit positional information associated with the nodes. In this type of graph, the connectivity and topology of the graph represent the primary information, while positional coordinates are not considered as features of the nodes.

- Example : Social Networks where nodes represent entities and relationships are the key information between the nodes. 

- Replace the function $\large f_{w}$ with 

    <img width="430" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/812f3e9f-d8c0-492a-889d-76cd6641b805">

    - Instead of concatenating all neighboring nodes into a single vector, it sums their values across all neighborhoods.

        <br/>

- Iteratively solve for $\large f_{w}$ and $\large o_{w}$ with given equations above, take a gradient step using Almeida-Pineda algorithm to minimize the objective loss function. 

- Repeat untill convergence.

<br/>


#### **1.1.2. Gated Graph Sequence Neural Networks (GGS-NNs)**

<br/>

- As described above, in order to ensure the existence of a unique solution after recurrent application of $\large f_{w}$, $\large f_{w}$ needs to satisfy the contraction mapping property.

- This requirement severly limits the modeling ability. 

- As an improvement of GNNs, GGS-NNs can remove the contraction mapping requirement by adding the gating mechanisms that controls the flow of information within the recurrent networks. 

- The gates, typically implemented as sigmoid or tanh functions, modulate the activation of the recurrent units and prevent unbounded growth or decay of the gradients.

- These adaptive behavior of gate can be an analogous to the contraction mapping that ensures stable updates of the recurrent states over long sequences. 

- Adaptaion 

    <img width="391" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/6e0c46b7-df34-40b0-b42f-581683e61b80">

- $\large z_{i}^{(t)}$ is calculated by the update gate.

<br/>

### **1.2. Graph-Level RNNs**

<br/>

- There have been other attempts to apply RNNs to capture graph-level patterns, including temporal patterns in dynamic graphs.

- [**You et al**](https://arxiv.org/abs/1802.08773){:target='_blank'} : 

    - GraphRNN learns to generate graphs by training on a representative set of graphs.

    - It divides the graph generation process into a sequence of node and edge formations using two RNNS, one for the nodes and the other for edges, conditioned on the graph structure generated so far.

-  **Dynamic graph neural network (DGNN)**

    - Utilized LSTM to learn temporal dynamics between nodes. 

    - Detailed explanation about this model is [**HERE**](){:target='_blank'}.

- **Graph RNN** combined with other structures such as **GCN** can be a good solution. 

    - **RMGCNN** : applied LSTM to GCN to progressively reconstruct a grpah. 

    - **Dynamic GCN** : applied an LSTM to gather the results of GCNs from different time slices in dynamic networks to capture both the spatial and temporal graph information.

 
<br/>

## **2. Graph Convolutional Networks (GCN)**

<br/>

- Aims to generalize convolutional networks (CNN) to graphs. 

    **Illustration of GCN**

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/395a9aa0-93a8-4218-8239-c2acc051734a" width="620">

<br/>
    
### **2.1. Convolution Operation**

<br/>

- There are largely two types of convolution methods used in GCN, one is spectral method and the other is spatial method.

<br/>

#### **2.1.1. Spectral Methods : Graph Fourier Transform**

<br/>

- Performs convolution by **transforming the node representation into the spectral domain** using fourier transform.

- **Fourier transform of graph signal** : **Eigen-decomposition of laplacian matrix** of the graph. 

    1. Fourier transform is a linear combinations of orthonormal bases. (here, each basis is a complex exponential function with different frequencies.)

    2. Finding orthogonal basis of a symmetric matrix with real eigenvalues can be done by eigenvector decomposition.

    &emsp; -> <span style='color:red'>If one can find a matrix associated to the graph signal and that matrix is real-symmetric, then fourier transform of the graph signal can be done by the eigen-decomposition of the matrix. </span>

    3. laplacian matrix represents captures the 
