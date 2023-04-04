---
layout: post
title : "[Neural Networks and Deep Learning] Deep Neural Networks (DNN)"
date: 2022-04-22 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---


<br/>

# **Deep Neural Networks (DNN)**
<br/>

## **What is Neural Network ?**
- every node recieves inputs and process them by it's own activation function and sends outputs to each node of next layer
- every layer has several nodes
- forward & backward propagation occurs for every optimization iteration
- <img src="https://user-images.githubusercontent.com/92680829/161899346-952dd9f6-5d3e-4428-98fe-21ad46997ebe.png"  width="600">
<br/>

- <img src="https://user-images.githubusercontent.com/92680829/161900602-c3199440-efd8-45bf-bd4b-8f22762c7787.png"  width="550">

<br/>

## **DNN notations**
- Input layer is called layer '0'
- L : Number of total layer
- n[i]n[i] : Number of units (nodes) of layer i
- a[i] : Activations (Outputs) of layer i -> a[i] = g[i] ( z[i] )
- w[i]  : Weights for z[i]
- b[i] : Bias parameter for z[i]

<br/>

## **Forward Propagation (FP)**
- Generalization : suppose we have only 1 node for each layer
    - propagating from [l-1] layer to [l] layer
    - each propagation incorporates 2 steps of calculation
        - computes Z[l]
        - computes a[l] = g(Z[l]), which is the final output of layer l
    - <img src="https://user-images.githubusercontent.com/92680829/161901638-5f8e6f54-2fe1-4ba9-990d-dec0190e378a.png"  width="200">

<br/>

### **Vectorized Implementation of Propagation through DNN with Multiple Nodes**
- now we have multiple nodes in each layer
- but it's too inefficient to compute all nodes by using for-loops
- Instead, we can **adapt vectorization by stacking all nodes of each layer into one Matrix**
- Example : 
    - <img src="https://user-images.githubusercontent.com/92680829/161908504-4901aeca-1c03-441e-870e-76f8b02e271e.png"  width="600">

<br/>

### **1. Vectorization of single example (x) and multiple nodes (p)**
- zi[1] (1x1) = wi[1]T (1x1) * x (nx1) + bi[1] (1x1) 
- a(zi[1]) : (1x1) matrix
- now stack up all nodes (p) in one Matrix vertically 
    - W[1] = (px1xn) matrix of all wi[1]T in layer 1
    - Z[1] = (p x 1) matrix

<br/>

### **2. Vectorization across multiple examples (m)**
- 1) Using For-Loop : suppose we have 2 layers 
    - <img src="https://user-images.githubusercontent.com/92680829/163290862-d12e1744-e4f9-4fbf-b61b-8eb6109588e1.png"  width="300">
   
- 2) Using Vectorization :
    - <img src="https://user-images.githubusercontent.com/92680829/163295299-f6117a86-26f7-4ade-8378-018af0690738.png"  width="250">

- **Justification for Vectorized Implemenation**
    - for simplification, assume that all b(i) equals to 0
    - <img src="https://user-images.githubusercontent.com/92680829/163297574-494cc42a-4a86-4184-9b40-5570db865376.png"  width="600">

- To wrap up, we can implement vectorization for propagation of multiple examples and nodes without using explicit For-Loop
    - X here can be alternatively expressed as A[0] (outputs of layer 0, which is input layer)
    -**Note that we can't eliminate for-loop for layers**
    - <img src="https://user-images.githubusercontent.com/92680829/163295410-609f3436-1722-4ed6-b87b-1eb0263a7c03.png"  width="250">

<br/>

## **Activation Functions**
- <img src="https://user-images.githubusercontent.com/92680829/163312344-00269c10-ea1d-486e-b81a-f1e5c99ffd38.png"  width="550">
- One of the most important thing to consider to improve algorithm is to choose activation function for hidden units as well as the output layer  
- Previously we've structured DNN algorithms by adapting sigmoid function of logistic regression
- Now, we'll gonna see other various options as activation functions

- Here are the nonlinear functions that is not sigmoid function (σ(z))
    1. **Tangent function** : tan(z)
    2. **Hyperbolic tangent function** : tanh(z) 
        - σ(z) : 1/1+e-z
        - tanh(z) : ez - e-z / ez + e-z
        - <img src="https://user-images.githubusercontent.com/92680829/163311098-b3e318ab-85b3-481a-b159-ba6659c1a163.png"  width="400">

<br/>

### **Tanh(z) vs σ(z) vs Relu**
- For almost all cases, **tanh(z)** always works better than **σ(z)** as an activation function, as it can center the data to have zero mean (**zero centered activation function**), not 0.5 mean, which makes next step of learning a lot more easier 
- One exception is with the output layer, which has the value either 0 or 1 (not -1 ~ 1)
- **Vanishing Gradient Problem (VGP)**
    - BUT, both activation functions have certain problem called VGP
    - The VGP occurs when the elements of the gradient (the partial derivatives with respect to the parameters of the NN) become exponentially small so that the update of the parameters with the gradient becomes almost insignificant 
    - This will severly slow down the speed of learning of neural network
    - To address this issue, one other option that is very popular in machine learning is **"Rectified Linear Unit"** called **"Relu"**

<br/>

### **Relu : max(0, z)**
- <img src="https://user-images.githubusercontent.com/92680829/163315014-9d77f242-0c65-439c-aef9-047e789c5bf7.png"  width="600">

- Advantages of relu 
    - Here, all the negative values are converted into 0 so there are no negative values available.
    - Maximum Threshold values are Infinity, so there is no issue of Vanishing Gradient problem (vgp) so the output prediction accuracy and learning efficiency are improved.
    - Speed is faster compared to other activation functions

- Advanced version of Relu : **Leaky Relu - max(0.01z, z)** 
    - Theere is one weakness of original Relu
    - for negative z, relu function only have zero value (no slope at all), so there is no learning power for the z with negative value
    - To resolve this, here is the Leaky Relu that has small slope (ai) for negtiave z  
    - slope coefficient is determined before training, not learnt during training
        - <img src="https://user-images.githubusercontent.com/92680829/163317464-b1795cb6-4b77-4556-b4c1-af8d9752aa12.png"  width="600">
        

- BUT **in practice**, enough of the hidden untis will have z greater than 0, so **original relu works just fine**

<br/>

### Rules of thumb for choosing activation functions
- Unless your output is either 0 or 1 for binary classification, tanh always works better than sigmoid function
- But both of them has VGP that can disturb proper learning 
- For now, Relu is increasingly the default choice of activation function

<br/>

### **Why do we need Non-Linear Activation function?**

- the composition of multiple linear functions is no more than another linear function
- You won't be able to compute any interesting functions (always linear functions with different W and b) no matter how you go deeper into neural networks
- So, it's pointless to combine and stack multiple hidden units 
- <img src="https://user-images.githubusercontent.com/92680829/163321384-9987a36b-1cee-45bf-80fb-f93e44ea8f78.png"  width="300">

- There is only one case where you can actually implement linear activation function
- **Output layer of linear regression problem** where y has real value 
- Even with linear regression problem such as housing price prediction, you can use other activation functions as well except linear fucntion
    - for the **hidden units**, you should alternatively use Relu or tanh function instead of linear function (works much faster and better)
    - for the **output layer**, you can actually use linear functions to drive actual price value

<br/>

## **Derivatives of Activation Functions**

<br/>

### 1. Sigmoid Activation Function
- σ(z)(1-σ(z))
- you can derive this with simple calculus
- <img src="https://user-images.githubusercontent.com/92680829/163323042-3ad50a63-83e8-4a30-b858-e621ffb9204b.png"  width="500">

- as z value go far from the center(0) either in a minus direction or plus direction
- derivative become closer to 0, which leads to vgp 

<br/>

### **2. Tanh Activation Function**
- 1 - tanh(z)^2
- As z value goes away far from 0, slope become flat, so derivative becomes 0, (you can see same result with the derivative formula of tanh above)
- at z value 0, derivative of tanh equals to 1
- <img src="https://user-images.githubusercontent.com/92680829/163324835-433d4d75-7074-4362-8f56-8fa3cdfb70b8.png"  width="450">

- **Derivation**
    - <img src="https://user-images.githubusercontent.com/92680829/163324633-481073fd-d4a7-4758-ac9c-359d85e63fc6.png"  width="500">

<br/>

### **3. Relu and Leacky Relu Activation Function**
- derivative of relu and leacky relu is very simple
- for Relu (when z=0, slope not defined)
    - if z < 0 : 0
    - if z > 0 : 1
- for Leacky Relu
    - if z < 0 : k (k is slope coefficient, in the example below, k = 0,1)
    - if z > 0 : 1

- <img src="https://user-images.githubusercontent.com/92680829/163326746-dc3848e0-cac5-47a6-a385-047dfb40b2b7.png"  width="600">

<br/>

### **Derivative of Multiple Activation Functions**
- here are the graphs and derivatives of various types of activation function
- <img src="https://user-images.githubusercontent.com/92680829/167841964-d9483851-6903-45a3-8d4f-c96b399ceb0e.png"  width="600">

<br/>

## **Gradient for Neural Network : Back-Propagation (BP)**
- to compute gradient descent and improve the parameters in NN, we have to use back-propagation
- It's just a mathmatical process to compute partial derivative of Error by desried parameter by adapting **Chain Rule**
- <img src="https://user-images.githubusercontent.com/92680829/163329926-707ee89a-d145-4418-87cf-e58adc573a26.png"  width="700">

- Let's define activation function as sigmoid
    - <img src="https://user-images.githubusercontent.com/92680829/164571040-92a6becb-d45d-4b4d-857f-ae9a1f591d4f.png"  width="400">
    
- Then, if you want to compute **partial derivative of L** by **w**, then you firstly have to compute it by **a**, then by **z**, and then finally by **w**, 
- Same applies to parameter **b**
    - <img src="https://user-images.githubusercontent.com/92680829/164571295-d51918ba-9d2f-4c60-add4-c6282b47bd85.png"  width="350">

<br/>

### --Calculations in detail--
- **dE/da = d(1/2*(Y-a)^2)/da = (Y-a)(-1) = a-Y
- <img src="https://user-images.githubusercontent.com/92680829/164571737-ce2cb9c1-4a41-4294-b8f9-f1ce06772c4f.png"  width="600">

<br/>

### **Vectorized Implementation of Back-Propagation**
<img src="https://user-images.githubusercontent.com/92680829/164575293-686d6309-1e5c-4ea9-a75e-3e1256a79447.png"  width="500">

<br/>

## **Random Initialization**
- For NN, it's important to set initial weights randomly unlike logistic regression where we can set all the first weight values by 0 
- if weights are equivalent each other for the first layer, all subsequent layers will end up receiving same values from previous layer, then all of your hidden units are symmetric.
- In this case, stacking up multiple hidden units for DNN becomes pointless as all of them will function exactly the same
- Otherwise, b is okay to be all same as 0 as it doesn't cause any symmetry issue

<br/>

### **Randomizing Parameters with Python**
- W[1] = np.random((n1, m)) * α  where n stands for the number of nodes at L1 and m is the number of training examples x
- b[1] = np.zeros((n1, 1)) 
- W[2] = np.random((n2, n1 (with k features)))
- b[2] = np.zeros((n2, 1)) 
- make sure **α has small values** such as 0.01
    - you can guess why by considering the feature of tanh or other sigmoid graph
    - for those activation functions, as the absolute value of x becomes large, the slope gets closer to 0, which can cause VGP
    - Thus if weights are too large, the output value of each node will become large as well
        - ai[j] =σ(w[j]a[j-1] + b[j]) 
        - as inputs to tanh are large, causing gradients to be close to zero. 
        - The optimization process will thus become slow 

- **Exercise)** 
    - the number of x is 4, and there are 3 nodes in first hidden layer 
    - what will be the vector shape of W[1], b[1], Z[1], A[1], respectively?
        - W[1] : (3,4)
        - b[1] : (3, 1)
        - Z[1], A[1] : (3,1)
