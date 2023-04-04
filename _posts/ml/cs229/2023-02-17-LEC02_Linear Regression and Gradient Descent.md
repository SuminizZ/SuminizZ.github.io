---
layout: post
title : "[Stanford CS229 01] Linear Regression and Gradient Descent"
img: ml/cs229.png
categories: [ml-cs229] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : truer
---
<br/>

## OUTLINES 

1. MULTIVARIATE LINEAR REGRESSION
2. BATCH/ STOCHASTIC GRADIENT DESCENT
3.  NORMAL EQUATION 

<br/>


---

## 1. Multivariate Linear Regression
<br/>

## **1.1. Multiple Features**
<br/>

- $x^{i}$ : $i_{th}$ input variables (set of features) <br>

- $y^{i}$ : $i_{th}$ output variable (target variable) that we're trying to predict <br>

- $ (x^{i}, y^{i})$ For $i = 1, 2, 3,...,m$ : training dataset <br>
<br>

- **Hypothesis**

    &emsp;&emsp;&emsp;&emsp;&emsp; $ h_{\theta}(x) = \sum \limits_{j=1}^{n} \theta_{j}x_{j} $ <br>

    - $x_{j}$ For $j = 1, 2, 3,...,n$ : value of $j_{th}$ feature of all n input features 
    - set $x_{0}$ as 0 (intercept term, b) <br>
    - $\theta_{j}$ For $j = 1, 2, 3,...,n$ : $j_{th}$ parameter of n parameters each  (weights) parameterizing the space of linear functions mapping from x to y <br> <br>

- **Matrix Representation of Hypothesis** <br>

    &emsp;&emsp;&emsp;&emsp; $\theta = \begin{bmatrix} \theta_{1} \\ \theta_{2} \\ . \\ . \\ . \\ \theta_{n} \end{bmatrix}$

    &emsp;&emsp;&emsp;&emsp; $ x^{i} = \begin{bmatrix} x^{i}_{1} \\ . \\ . \\ . \\ \end{bmatrix}$

    &emsp;&emsp;&emsp;&emsp; $h(x^{i}) = \theta^{T}x^{i}$

<br>

## **1.2. Cost Function**
<br/>

- Trying to minimize the deviations of $h(x)$ from $y$ <br>
- Least Mean Square (LMS algorithm) <br>

&emsp;&emsp;&emsp;&emsp; $ J(\theta) = \sum\limits_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2} $
  

- **LMS algorithm with Gradient Descent**
    - algorithm starts with some initial guess with $\theta_{j}$ with radomized values and repeatedly updates the paratmeters using gradient descent algorithm <br> 
    - take partial derivative with respect to every parameter multiplied by learning rate ($\alpha$) and substract it from previous value of paramter <br>
    
    &emsp;&emsp;&emsp;&emsp; $ \theta_{j} := \theta_{j} - \alpha\frac{\partial J(\theta)}{\partial \theta_{j}}$ &emsp; $For j = 1,2,3,...,n $ 
    <br>
    - $\alpha$ (learning rate) : regulates the speed of adjusting parameters so that prevents over-fitting 
        - try multiple cases and find best one
    - repeat updating parameters for every step of gradient descent <br>
    
     - **Partial Derivative of $J(\theta)$**
         <img src="https://user-images.githubusercontent.com/92680829/226507002-653a8d8b-8c7c-443e-a81d-18e4b0076a7e.png" width="400">
     
     - $\theta_{j} := \theta_{j} - \alpha (h_{\theta}(x) - y)x_{j} $ &emsp; $ For j = 1,2,3,...,n $
     - larger change will be made with larger error term ($ h(\theta) - y $) 
     - Repeat the update untill **convergence** 

<br/>

---

## **2. Batch Gradient Descent (BGD) vs Stochastic Gradient Descent (SGD)**

<br/>

- In **BGD** (1 update per batch):
    - the algorithm updates the model parameters after processing the entire training dataset. 
    - The cost function $ J(\theta) $ is first computed over all the training examples and then the gradient of the cost function with respect to the parameters is computed. <br>
    &emsp;&emsp;&emsp;&emsp; $ \theta_{j} := \theta_{j} - \alpha (h_{\theta}(x) - y)x_{j} $ for every j
    <img src="https://user-images.githubusercontent.com/92680829/226516910-a83cf250-f717-47b4-8197-79227691fc6c.png"  width="330">
    
<br/>

- In **SGD** (1 update per data point): 
    - updates the model parameters after processing each individual training example. 
    - for each iteration, the algorithm randomly selects one training example, computes the gradient with respect to that example, and then updates the parameters based on that gradient.
    <img src="https://user-images.githubusercontent.com/92680829/226513938-b576bc1f-6352-457b-a37a-cd207faee8c0.png" width="450">
    <img src="https://user-images.githubusercontent.com/92680829/158748465-5e302586-7b60-4960-b7a4-bd43a121bbce.png"  width="330">

<br/>
    
- BGD processes the entire training set at each iteration, which is computationally expensive but accurate. 
- SGD processes a signle training example at a time so that can coverge much faster. 
- While SGD has economical advantage over BGD, it may never be converge on global minimum, only oscillating around the local minimum. 
- Therefore, BGD can converge to the optimum more accurately and quickly on small datasets, while SGD can converge faster on large datasets.

<br/>

### **Check for Convergence with Stochastic gradient descent**

<br/>

- how to check SGD has convergd to global minimum (at least close)
- how to tune learning rate α to get proper convergence? 

- **Plotting $ J(\theta) $ averaged over N examples**
    1. decrease learning rate (upper left)
        - slower the convergence
        - but obtain slightly better cost (negligible sometimes)
    2. increase N (>= 5000) (upper right)
        - also takes more time to plot (longer time to get single plotting point)
        - can smoothen the cost line
    3. increase N (lower left)
        - line will fluctuate too much, preventing you from seeing actual trend 
        - if you elevate N, then you can see what's actually going on 
    4. decrease learning rate (lower right)
        - it shows that your algorithm fails to converge to minimum, (diverging, fails to find optimal parameters)
        - you should adjust your learning rate smaller, so that it can converge
    
    <br>
    
    <img src="https://user-images.githubusercontent.com/92680829/158756682-e98ca189-c71e-48a8-929a-9718bbb3967b.png"  width="500" >
        
<br/>

- **Learning rate (α)**
    - typically, α helds constant through entire learning process
    - but, can also slowly decrease α over time (if you want the model to converge better)
        - **α = $\beta\,$ / ($\,$iterationNumber$\,$ + $\gamma$)**
        - need to take additional time to decide what $\beta$ and $\gamma$ are
        - guaranteed to converge somewhere rathter than oscillating around it

- SGD can be a good algorithm for massive training examples

<br/>

---

<br/>

## 3. Normal Equation 

- Closed-form solution for linear regression problems, which can be used to find the optimal parameters of a linear model. (only applicable to linear regresson case)
- provides a way to compute the optimized parameter vector theta directly from the training data by solving the equation, without the need for an iterative optimization algorithm such as gradient descent.
- Explicitly takes derivatives of cost function with respect to $\theta_{j}$s and solve by setting it to be 0. 

<br/>

### 3.1. Matrix Dervatives 

<img width="500" alt="Screen Shot 2023-03-22 at 10 20 37 PM" src="https://user-images.githubusercontent.com/92680829/226919657-9eb90839-890c-4153-a231-8b98cc2b3daf.png">


<br/>


### 3.2. Properties of $\nabla$ and Trace of Matrix 

<img width="594" alt="Screen Shot 2023-03-22 at 10 20 42 PM" src="https://user-images.githubusercontent.com/92680829/226920487-f70958cd-339e-44b5-b51f-79a4eb51d8cf.png">

<br/>


### 3.3 Least Mean Square solved with Normal Equation 

<img width="720" alt="Screen Shot 2023-03-22 at 10 20 47 PM" src="https://user-images.githubusercontent.com/92680829/226920887-661939ea-7392-4103-86df-d515270b3126.png">

- The amout of computations needed to solve normal equation depends on n (the number of parameters) with $O(n^{2})$
- For dataset with smaller number of paramters, solving normal equation instead of iterative gradient descent will be efficient.
