---
layout: post
title : "[Paper Review] Inductive Representation Learning on Large Graphs (GraphSAGE, 2017)"
img: papers/gnn/gsage.png
categories: [papers-gnn]  
tag : [Paper Review, GNN, GCN, ChebNet]
toc : true2020
toc_sticky : true
---

## **Outlines**

- [**Reference**](#reference)
- [**1. Spectral Convolution**](#1-spectral-convolution)
- [**2. ChebNet : Re-Formulation of Spectral Filtering with K-Hops Localized Filters**](#2-chebnet--re-formulation-of-spectral-filtering-with-k-hops-localized-filters)
    - [**2.1. K-Localized Spectral Filters**](#21-k-localized-spectral-filters)
    - [**2.2. Fast and Stable Filtering : Recursive Formulation of Chebyshev Polynomials**](#22-fast-and-stable-filtering--recursive-formulation-of-chebyshev-polynomials) 
    - [**2.3. Learning Filters**](#23-learning-filters) 
    - [**2.4. Group Coarsening and Fast Pooling**](#24-group-coarsening-and-fast-pooling) 
- [**3. GCN : Layer-Wise Linear Formulation of Spatial Graph Convolution**](#3-gcn--layer-wise-linear-formulation-of-spatial-graph-convolution)
    - [**3.1. Layer-Wise First-Order Linear Model**](#31-layer-wise-first-order-linear-model)
    - [**3.2. Single Parameter**](#32-single-parameter)    
    - [**3.3. Renormalization Trick**](#33-renormalization-trick)    
    - [**3.4. Multi-Layer GCN for Semi-Supervised Learning**](#34-multi-layer-gcn-for-semi-supervised-learning)    

<br/>

## **Reference**

<br/>

- [**Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, Defferrard et al, 2016**](https://arxiv.org/pdf/1606.09375.pdf){:target="_blank"}
- [**Semi-Supervised Classification with Graph Convolutional Networks, Kipf et al, 2017**](https://arxiv.org/pdf/1609.02907.pdf){:target="_blank"}
- [**Spectral GCN**](https://tootouch.github.io/research/spectral_gcn/){:target="_blank"}
- [**ChebNet PyTorch Implementation**](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/Chebnet_Blog%2BCode.ipynb){:target="_blank"}
- [**[논문 리뷰] Graph Neural Networks (GCN, GraphSAGE, GAT)**](https://www.youtube.com/watch?v=yY-DpulpUwk){:target="_blank"}

<br/>