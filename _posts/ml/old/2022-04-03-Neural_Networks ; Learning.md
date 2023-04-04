---
layout : post
title : "[Coursera : ML Specialization] - Neural Representation : Learning"
date: 2022-04-03 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-ml] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>


## **Neural Networks : Classification**
<br/>

<img src="https://user-images.githubusercontent.com/92680829/156963145-e4a1b901-d390-4aee-8535-6be459d6169a.png" width="600">
<br/>

### **Cost Function for Neural Network with Sigmoidal Activation**
<br/>

- The (regularized) logistic regression cost function is as follows;
    - <img src="https://user-images.githubusercontent.com/92680829/156687790-4532412e-706c-435c-b5aa-7d4a5f9145c3.png" width="600">


- For neural networks, regularized cost function is a generalization of this equation above (instead of one output, we generate k outputs)
    - <img src="https://user-images.githubusercontent.com/92680829/156964377-a4b5d0f8-d17a-4982-8ba4-438b4cb676bf.png" width="900">
    - finally, output is class that can maximize the probability that h(x) = 1

<br/>



## **Back Propagation Algorithm : to minimize cost function**
<br/>

- need to compute **1. J(Ɵ)** and **2. derivative of J(Ɵ)**
- Partial derivative of J(Ɵ)
    - Ɵ is indexed in three dimensions because we have separate parameter values for each node (j) in each layer (L) going to each node (i) in the following layer
    - from j (in layer L) to i (in layer L+1)
    - each layer has a Ɵ matrix associated with it
    - We want to calculate the partial derivative of Ɵ with respect to a single parameter 
    - <img src="https://user-images.githubusercontent.com/92680829/156965682-06c60ecd-8f30-4022-8d39-e4940990efb7.png" width="150">

<br/>

### **Forward propagation algorithm** operates as follows**
 <br/>

    - Layer 1
       - a1 = x 
       - z2 = Ɵ1a1
    - Layer 2
        - a2 = g(z2) (add a02), g here is sigmoidal function
        - z3 = Ɵ2a2
    - Layer 3
        - a3 = g(z3) (add a03)
        - z4 = Ɵ3a3
    - Output
        - a4 = hƟ(x) = g(z4)

<br/>


### **Backward propagation algorithm**
<br/>

- to compute the partial derivatives
- For each node we can calculate (δjl) - this is the error of node j in layer l
    - we need to calculate error compared to the "real" value
    - the only real value we got is the result (actual y), so we have to **start with the final output!!**

    - <img src="https://user-images.githubusercontent.com/92680829/156968787-5607b197-68f3-412f-88fb-6765357f2184.png" width="300">
    
<br/>

- Given the above neural network as an example,
- first error : 
    - δj4 = aj4 - yj
    - [Activation of the unit] - [the actual value in the training set] (aj4 = hƟ(x)j)
   
- derivative 
    <img src="https://user-images.githubusercontent.com/92680829/156974170-fdcf87d4-78a0-40df-abfb-5b7d13ab2b47.png" width="250">

    - out (aj[L]) here is current node (aj[L])  : layer L  </br>
    - **delta(δ)** here is from [ (ai[L+1] - y) * (ai[L+1])(1-ai[L+1]) ]  : layer L+1  
    
<br/>

### **Summary**
<br/>

- loop through the training set

- Set a1 (activation of input layer) = xi 
    - Perform forward propagation to compute a(L) for each layer (L = 1,2, ... L)
    - run forward propagation
    - Then, use the **output label** to calculate the delta value for the output layer δL = aL - yi)
    - Then, using back propagation we move back through the network from layer L-1 down to layer 
    - Finally, use Δ to accumulate the partial derivative terms

        <img src="https://user-images.githubusercontent.com/92680829/156976785-882bd053-cc84-45c5-8892-6fa77d49e721.png" width="300">
 <br/>
       
    - Note that
        - l = layer
        - j = node in that layer
        - i = the error of the affected node in the target layer


<br/>

### **-- proof of derivative of J(Ɵ) --**
<br/>

[if you want to know more detailed process of derivation, click here](https://goofcode.github.io/back-propagation)
- with **Chain Rule**

<img src="https://user-images.githubusercontent.com/92680829/156974547-774345eb-1ca5-44f9-b6de-741b29071138.png" width="350">
<br/>

- (1) ai[L+1] - y
- (2) (ai[L+1]) * (1-ai[L+1])  : derivative of sigmodial function of z (net)
    - δi[L+1] = (1) * (2) = (ai[L+1] - y) * (ai[L+1])(1-ai[L+1])
- (3) aj[L]
- (1) * (2) * (3) = updating value of W

<img src="https://user-images.githubusercontent.com/92680829/156972765-21ffaccb-7fc0-4096-92af-bb0b891fd8ab.png" width="450">
<br/>



- So we need δ to update the W (theta here), then 
- **How to get δ** : Back-Propagation

<img src="https://user-images.githubusercontent.com/92680829/158723928-1ae750bf-da88-492d-951a-7ebebf167919.png" width="600">

<br/>


## **BackPropagation Algorithm Intuition**
<br/>

### **Forward Propagation to get activation values of each layer**
<br/>

<img src="https://user-images.githubusercontent.com/92680829/156980584-7dd139ce-5f8d-46a8-af0c-fc7cb0917cff.png" width="700">

- 1. get z (each x multiplied by Ɵ)
- 2. calculate g(z) (in this case, sigmoidal function)

<br/>

### **Back Propagation to minimize the cost function**
<br/>

- δ term on a unit as the "error" of cost for aj(L) (unit jth in layer L)
- **δi[L+1] = (ai[L+1] - y) * (ai[L+1])(1-ai[L+1])**
- J(Ɵ) = cost(i)
- z : net

- <img src="https://user-images.githubusercontent.com/92680829/156982115-5feb0f67-1e09-4cfd-a7ae-60eb8dcb5a9a.png" width="200">


- **What is Back propagation?** 
    - it calculates the δ, and those δ values are the weighted sum of the next layer's delta values, weighted by the parameter associated with the links
    - δ2(2) = [Ɵ12(2) * δ1(3)] + [Ɵ22(2) * δ2(3)] + [Ɵ32(2) * δ3(3)]
    - <img src="https://user-images.githubusercontent.com/92680829/156984818-d43c4172-2d87-435c-a0a7-d16d284f62a9.png" width="700" >

<br/>

## **Gradient Checking**
<br/>

- back prop algorithm is quite susceptible to many subtle bugs, which leads you to get a higher levels error than you do in bug-free implementation
- there is a good way to deal with most issues associated with the buggy back-prop, called **Gradient Checking**
- get Numerical Approximates of Gradient
    </br>
    - <img src="https://user-images.githubusercontent.com/92680829/156989063-78e4ee34-3f47-40b3-bf05-7df01685c238.png" width="600">
    - <img src="https://user-images.githubusercontent.com/92680829/156990466-c6f7f3c7-797c-4e26-a9a7-7149ea1b4fd8.png" width="500">

<br/>

- then, check whether approximates have similar value with derivative gained from back-prop

<br/>


## **Random Initialization**
<br/>

- Pick random small initial values for all the theta values of first input layer
    - If you start them on zero (which does work for linear regression) then the algorithm fails - all activation values for each layer are the same
    - if you set all theta by an identical number, you'll faill to break the symmetry, as all hidden units will repeatedly get exactly the same signal from previous layers
- So chose random values to break the symmetry!! 
    - Between 0 and 1, then scale by epsilon (where epsilon is a constant)
    - rand(10, 11) (10x11 vector) * 2(init_epsilon) - (init_epsilon)
        - [-epsilon, +epsilon]

<br/>


## **Putting it all together**
<br/>

<img src="https://user-images.githubusercontent.com/92680829/156998625-cca147dc-f987-49da-a646-53c22354404b.png" width="600">

- Use gradient descent or an advanced optimization method with back propagation to try to minimize J(Ɵ) as a function of parameters Ɵ
    - But, J(Ɵ) is non-convex --> could be susceptible to local minimum
    - In practice this is not usually a huge problem
    - Can't guarantee programs to find global optimum, instead, should find **good local optimum at least**
    <img src="https://user-images.githubusercontent.com/92680829/157000799-64e649d9-73e5-44ad-b1e1-43daf5fb5a1a.png" width="600">
