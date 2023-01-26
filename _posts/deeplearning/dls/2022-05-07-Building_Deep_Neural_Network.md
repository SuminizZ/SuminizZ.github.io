---
layout: post
title : "[Neural Networks and Deep Learning] Building Deep Neural Network Step by Step"
date: 2022-05-07 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---

<br/>

## - Vectorized Implementation for Propagation
- As we've seen from the shallow neural network that we had built [<span style="color:blue">**previously**</span>](https://suminizz.github.io/deep%20learning/Practice-Planar-data-classification-with-a-hidden-layer/){:target="_blank"}, vectorized implemantation allows us to propagate through multiple examples at a time without using explicit for-loop in algorithm, which can save the learning time significantly.
- When implementing vectorization for your model, making sure that the dimensions of matrices that are used to be consistent is really important for debugging 
- Generalization 
    <img src="https://user-images.githubusercontent.com/92680829/169811495-4b42821a-790d-4b1d-8f9b-b779f73fee5f.png" width="400">
- Keeping straight the dimensions of various matrices and vectors you're working with will help you to eliminate some classes of bugs in your model. 

<br/>

## - Why Deep Representation?
- We've all been hearing that neural networks that are **deep** (with lots of hidden layers) work better than the ones with shallow representation.
- But WHY is it so? 
- For this chapter, let's go through a couple of examples to gain some intuition for why **deep is better than shallow**

<br/>

### 1. Face recognition
<img src="https://user-images.githubusercontent.com/92680829/169816264-089b7178-e317-43df-b2e1-b4041f2783d9.png" width="630">
- Suppose you have an face recognizing algorithm with 20 hidden layers
- If you input a picture of a face, then the first layer will act as somewhat a feature detector or edge detector (will be dealt in depth at later courses about CNN) by grouping together the pixels to form edges
- 20 hidden layers might then be trying to figure out the orientations of those edges in the image to structure the image horizontally and vertically and group the edges to form small parts of a face
- As we go deeper down the layers of model, finally by putting together each different part of a face, like an eye or a nose, it can then try to recognize or even detect different types of faces 
- So intuitively, you can think of the **earlier layers** of the neural network as **detecting simple functions**, like edges. And then composing them together in the **later layers** of a neural network so that it can **learn more and more complex functions**

<br/>

### 2. Circuit Theory
- There are functions you can compute with **Small L-layer Deep** neural networks which, otherwise, **shallower networks require exponentially more hidden units** to compute
- Suppose you're trying to compute an exclusive OR (XOR) problem for n x features (x1 XOR x2, x2 XOR x3, x3 XOR x4 ... )
    - the depth of the network to build XOR tree for n x features will be on the order of log n (O(logn))
        - you only need to iterate calculations for log2n here (technically, you need a couple layers to copmute 1 XOR function - (x1 + x2)*(not x1 + not x2))
        - but still it's a relatively small circuit (still complexity is O(logn))
        <img src="https://user-images.githubusercontent.com/92680829/172041734-118a11ce-47b4-49b5-93b5-b1323b9d9393.png" width="400">
    - But if you're not allowed to use neural network with multiple layers, then you need 2^n units because you need to enumerate 2^n possible combinations (O(2^n)). 
        - 2 units needed for one x feature
    
    <img src="https://user-images.githubusercontent.com/92680829/169826218-6536ab52-eaad-45f7-b48a-6b4fc2978d8d.png" width="480">
    - This shows that deep hidden layers allow you to compute exactly the same funciton with relatively smaller hidden unit size compared to the shallow neural networks
    - Large unit size requires more calculations, which significantly lowers the learning efficiency of an algorithm

<br/>

## - Basic Building Blocks for DNN : FP & BP
- repeat **forward propagation** and **backward propagation** untill it reaches the global optimum
<img src="https://user-images.githubusercontent.com/92680829/172043492-5b0efa18-bd58-45a8-be4e-b0ca7d9e8cbe.png" width="700">
    - **Forward Propagation**
        - <img src="https://user-images.githubusercontent.com/92680829/172045925-a4cbc7e3-c85f-4663-b9ae-2bc2f1996e7e.png" width="180">
    - **Backward Propagation**
        - <img src="https://user-images.githubusercontent.com/92680829/172045956-70884055-6eda-4f40-ba72-b7c32cc6a1ba.png" width="420">

<br/>

- **Summary of whole FP, BP process**
<img src="https://user-images.githubusercontent.com/92680829/172048023-26e9fe85-3818-4c0c-81f5-e522f913e0f4.png" width="1000">

<br/>

## - Parameters vs Hyper-Parameters 
- **Parameters** : w[l], b[l]
    - these parameters are learnt through the learning process of DNN such as Gradient Descent
- **Hyperparameters** : learning rate (α), # of iterations, # of hidden layers, hidden unit size for each layer, chocie of activation function, momentum, minibatch size, regularization parameters ... etc.
    - for the case of hyperparameters, its not something that you can learn through algorithm
    - its just somthing that you should choose empirically by applying every appropriate combinations of hyperparameters 
    - empirical process : a fancy way of saying that you try out a lot of things and figure out the best options just like you're doing somewhat experiments 
