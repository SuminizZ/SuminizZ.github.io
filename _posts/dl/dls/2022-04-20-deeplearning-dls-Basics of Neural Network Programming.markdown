---
layout: post
title: "[Neural Networks and Deep Learning] Basics of Neural Network Programming"
date: 2022-04-20 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dl-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---

<br/>

## **Binary Classification**
<br/>

-  Example : Cat Classifier
    - with a given image, you can convert the image to three 64 x 64 matrices corresponding to Red, Green, Blue pixel intensity values for your image
    - Now unroll all these matrices to a single feature vector X, with the size of [64x64x3 x 1] vector 
    - <img src="https://user-images.githubusercontent.com/92680829/159686660-02d7def4-1739-4e30-bbd3-697f96abb98e.png"  width="550">

<br/>

- Notation
    - stacking different data(examples) into different column of X and Y
    - <img src="https://user-images.githubusercontent.com/92680829/159687336-06bad1e9-b864-4a29-9f38-4d24226e6975.png"  width="500">

<br/>

- X.shape = (nx, m)
- nx : length of x(i), the size of all R, G, B matrices unrolled
- Y : [y1, y2, y3..., ym] (1, m) 

<br/>

## **Logistic Regression as a Neural Network**
<br/>

- Binary output : Outputs of Y is always either 1 or 0
    - you want
    - <img src="https://user-images.githubusercontent.com/92680829/159688518-5dc9fc6a-ab5e-4b2b-bd08-b99927d1be23.png"  width="350">


- **In Linear Regression**
    - you can get output by using the equation **y = WTx + b**
    - W : (nx, 1) vector of weights of each feature / b : real number (intersect)
    - BUT, with this linear function, you can't get what you want, the chance that y of given example equals to 1 (value ranging from 0 ~ 1)

- In Logistic Regression
    - Instead, you can use sigmoid function with which you can get the output ranging from 0 ~ 1 depending on the x values
    - <img src="https://user-images.githubusercontent.com/92680829/159690058-5a5af644-c928-499a-a494-911aeb44741b.png"  width="250">


- here, z equals to the previous value obtained from linear regression, WTX + b
- <img src="https://user-images.githubusercontent.com/92680829/159690511-892b5dc6-e079-44df-b7c7-5139d49ed710.png"  width="400">

- when x infinitely increases, g(z) converge to 1, whereas x infinitely decreases, g(z) converge to 0.
- all g(z) values are within between 0 ~ 1
- when z equals to 0, you will get 0.5 as g(z)
- Also, you can alternatively define **b** as **x0** and set w0 as 1, so that you can incorporate b into WTX part
    - here's the outcome
    - <img src="https://user-images.githubusercontent.com/92680829/159691164-fe46ef04-02a0-4bad-806b-8214f0bc1da5.png"  width="150">


<br/>

### **Logistic Regression Cost Function**
<br/>

- Training set of m training examples, Each example has is n+1 length column vector
- <img src="https://user-images.githubusercontent.com/92680829/156683168-6dfb6801-f65a-4a2c-815a-4d37f69839a8.png" width="500" >

- Given the training set how to we chose/fit θ?
    - Cost function of linear regression was like below, 
    - <img src="https://user-images.githubusercontent.com/92680829/156683283-033fa772-f636-4bdf-87f3-643d477483a1.png" width="300" >

- Instead of writing the squared error term, we can write 
- **cost(hθ(xi), y) = 1/2(hθ(xi) - yi)2**
- Which evaluates the cost for an individual example using the same measure as used in linear regression
    - We can redefine J(θ) as
    - <img src="https://user-images.githubusercontent.com/92680829/156683371-e8fb4778-a11f-4199-a99e-3ca4de1588fa.png" width="300" >

Which, appropriately, is the sum of all the individual costs over the training data (i.e. the same as linear regression)

- This is the cost you want the learning algorithm to pay if the outcome is hθ(x) and the actual outcome is y
- Issue : If we use this function for logistic regression, this is a **Non-convex function** for parameter optimization
    - non-convex function : wavy - has some 'valleys' (local minima) that aren't as deep as the overall deepest 'valley' (global minimum).
    - Optimization algorithms can get stuck in the local minimum, and it can be hard to tell when this happens.
 - **A convex logistic regression cost function**
     - To get around this we need a different, convex Cost() function which means we can apply gradient descent
     - <img src="https://user-images.githubusercontent.com/92680829/156684011-edda5943-64ce-43b9-924b-7a7fd1ce0ddc.png" width="400" >

- This is our logistic regression cost function
    - This is the penalty the algorithm pays
    - Plot the function

    1. Plot y = 1
        - So hθ(x) evaluates as -log(hθ(x))
        - <img src="https://user-images.githubusercontent.com/92680829/156685913-f0e750ef-56db-4deb-9a0d-f2cbcab3e3f5.png" width="220" >
        
    2. plot y=0
        - So hθ(x) evaluates as -log(1-hθ(x))
        - <img src="https://user-images.githubusercontent.com/92680829/156686219-e35d4c6c-2001-480a-9acd-cd927f906fb3.png" width="220" >
    
<br/>

### **Combined Cost Function of RL**
<br/>

- Instead of separating cost function into two parts differing by the value of y (0 or 1),
- we can compress it into one cost function, which makes it more convenient to write out the cost.

    - **cost(hθ, (x),y) = -ylog( hθ(x) ) - (1-y)log( 1- hθ(x) )**
    - y can only be either 0 or 1
    - when y = 0, only -log( 1- hθ(x) ) part remains, which is exactly the same as the original one
    - when y =1, only -log( hθ(x) ) part remains
    - <img src="https://user-images.githubusercontent.com/92680829/156687790-4532412e-706c-435c-b5aa-7d4a5f9145c3.png" width="600" >

- now! you can finally get convex cost function that has global optima

<br/>

### **Optimizing Cost Function w/ Gradient Descent**
<br/>

- Interestingly, derivative of J(θ) of logistic regression is exactly identical with that of linear regression (proof of this statement will be covered later)
- Firstly, you would set all the features(w1~wm) as 0, including w0 (intersect, b) 
- and then, Repeat
    - <img src="https://user-images.githubusercontent.com/92680829/156696635-ab555f91-5544-40e9-9855-fe92787b3901.png" width="350" >
- Representation of the process of finding global optima
    - <img src="https://user-images.githubusercontent.com/92680829/160386110-1938217c-1d25-4455-a643-f49c84816f51.png"  width="500">

- BUT! this optimizing algorithm has serious weakness, which is explicit double for-loop
- first for-loop is for iterations of algorithm untill you reach to global optima
- secondly, you need to have a for loop over all the features
- this explicit for-loop can severly slower the training rate with the large dataset
- So, instaead of this, you need to learn **"Vectorization"** with which you can get rid of these explicit for-loop

<br/>

#### **-- Proof : Getting Derivative of LR Cost Function --**
<br/>

- Remember hθ(x) is
    - <img src="https://user-images.githubusercontent.com/92680829/160383571-e315c407-ec0f-4e69-95cb-b30d669d4435.png"  width="200">

- Step1 : take partial derivative of h(θ) = 1/(1 + e-z) 
    - <img src="https://user-images.githubusercontent.com/92680829/156696402-799da3b1-8d66-4ab4-b7e6-8e92c27d46f3.png" width="400" >

- Step2 : take partial derivative to J(θ)
    - <img src="https://user-images.githubusercontent.com/92680829/156696529-a943aceb-f987-4324-9a57-b2ad41e3a35f.png" width="400" >
    - <img src="https://user-images.githubusercontent.com/92680829/160385000-1694ccca-9f27-413d-a8fc-bd15794a7237.png"  width="400">       
    - <img src="https://user-images.githubusercontent.com/92680829/156696592-9857ffd5-6637-46ed-abef-2f47d21b64c0.png" width="500" >

<br/>

### **Computation Graph**
<br/>

- Previously, I figured out the partial derivative of J (dJ/dθ), by using **Chain Rule**
    - Chain Rule : backward propagation of taking derivative partially with respect to from final output variable (here, v) to starting variable (here, a)
    - <img src="https://user-images.githubusercontent.com/92680829/160389908-ce557dd3-7a18-40ed-a52d-6816c0addfc2.png"  width="500">
    
<br/>

## **Vectorization with Python**
<br/>

- vectoriztion can save you a great amount of time by removing explicit for loop from your algorithm!
    - <img src="https://user-images.githubusercontent.com/92680829/160834341-2ff411f3-9cc5-4b3a-98c3-c2d8a2849fa2.png"  width="500">
        
    - let's see if it's true with python code


```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
vec= np.dot(a, b)    # calculate inner product of a, b vector (1D)
toc = time.time()

print(vec)
print("Vectorized Version : {0}ms".format(1000*(toc-tic)))

tick = time.time()
skr = 0
for i in range(1000000):
    skr += a[i]*b[i]

tock = time.time()
    
print(skr)
print("Scalar Version : {0}ms".format(1000*(tock-tick)))
```

    249812.28927442286
    Vectorized Version : 2.006053924560547ms
    249812.28927442944
    Scalar Version : 1888.9873027801514ms


- the results of both algorithm are same
- BUT, it takes about 1000 times longer time to calculate the inner product of 1d vector a & b

- There are some numpy functions that allow you to apply exponential or log operation on every element of a matrix/vector
- np.log(V), np.exp(V)

<br/>

### **Logistic Regression with Vectorization**
<br/>

- logistic regression with For-Loops
    - suppose we have 'n' features
    - there are 'm' samples 
    - without vectorization, you have to use 2 for-loops, one for i (from 1 to n) and another for j (from 1 to m)
    <img src="https://user-images.githubusercontent.com/92680829/161545636-50b6f7b2-a508-4faf-a177-ab4a0a3732c5.png" width="460" >

- **Vectorizing Logistic Regression**
    - with vectorized LR, all you need to calculate the Gradient Descent of Cost function for each iteration is just two liens of code
    - db = 1/m(np.sum(dZ)
    - dw = 1/m(XdZT)
    - <img src="https://user-images.githubusercontent.com/92680829/161550157-e994fe73-0a1f-49af-a24d-f0adc4f94752.png"  width="270">
        
    - you don't need **"ANY"** foor-loops
    - but even with vectorized LR, you still need to use for-loop for iterations to gd minimizing the cost 
    <img src="https://user-images.githubusercontent.com/92680829/161549261-0fba8b93-f524-4b90-8355-756ed790e51c.png"  width="220">

<br/>

### **Broadcasting in Python**
<br/>

- It refers to how **numpy** treats arrays with different Dimension during arithmetic operations(+, -, *, /) which lead to certain constraints
- the smaller array is broadcasted across the larger array so that they have compatible shapes
- <img src="https://user-images.githubusercontent.com/92680829/161551568-dba2c92a-53bb-4ca6-9bf1-d04dc6cc50bd.png" width="410" >

- Even though broadcasting of python-numpy provides lots of benefits such as convenience and flexibility, but it can also cause a few bugs when mistreated
- For effective usage of only the strengths of broadcasting of python-numpy, except the weaknesses
- recommend not to use "Rank 1 Array" like np.random.randn(5), which has pretty non-intuitive shapes
- Instead, you can use vector like np.random.randn(5,1)
