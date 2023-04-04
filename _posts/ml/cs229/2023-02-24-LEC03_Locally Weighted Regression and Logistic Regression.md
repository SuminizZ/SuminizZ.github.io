---
layout: post
title : "[Stanford CS229 02] Locally Weighted Regression and Logistic Regression"
img: ml/cs229.png
categories: [ml-cs229] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : truer
---
<br/>

## OUTLINES

1. Locally Weighted Regression
2. Probabilistic Interpretation (Maximum Log Likelihood)
3. Logistic Regression
4. Newton's method
<br/>


---
<br/>

# 1. Locally Weighted Regression

- fitting a model to a dataset by giving more weight to the data points that are close to the point being predicted
- Non-parametric learning algorithm where the number of parameters you need to keep grows with the size of the dataset, while parametric learning has fixed set of parameters.

<br/>

## 1.1. Cost function to minimize 


&emsp;&emsp;&emsp;&emsp; $\normalsize \sum\limits_{i=1}^{m} \omega^{i}(y^{i} - \theta^{T}x^{i})^{2}$ &emsp; where &emsp; $\normalsize  \omega^{i} = exp(\frac{- (x^{i} - x)^{2}}{2}) $ <br>
    
- Weighting function $\omega^{i}$ : used to assign a weight to each training example based on its distance from the point being predicted.
- $x^{i} \,$ : data points that you're processing 
- $x \,$ : point of interest to be predicted
- Automatically gives more weight to the points of close to $x$ (max weight = 1)
- Points too far from the point of interest will fade away with infinitesimally small weight $\omega^{i}$
- locally fit an almost straight line centered at the point to be predicted. <br>
<br/>

<img src="https://user-images.githubusercontent.com/92680829/227078685-2665a976-5c46-42af-a9a8-893d3c074a8c.png" width="570">

## 1.2. $\normalsize \tau$ : bandwidth parameter
<br/>

&emsp;&emsp; $\large \omega^{i} = exp(\frac{-(x^{i} - x)^{2}}{2\tau^{2}})$

- Weight term depends on the choice of $\large \tau$
- this controls how quickly the weight is adjusted by the distance of data points from the point to be predicted. 
- called as bandwith parameter as it determines the width of linearly fitted local area with respect to the query point. 

<br/>

---
<br/>

# 2. Probabilistic Interpretation of Least Mena Square
<br/>

- Conver the problem from ``minimizing error term`` to ``maximize the probability`` of $y^{i}$ given with $x^{i}\,$ parameterized by  $\theta$
- Can make an assumption that $\epsilon^{i}$ are distributed IID (independently and identically distributed)
- According to the Central Limit Theorem (CLT) with large enough training examples, $\epsilon^{i}$ converges to Gaussian Distribution <br>

    &emsp;&emsp; $\normalsize \epsilon^{i} \sim~ \mathcal{N}(\mu = 0,\,\sigma^{2})\,$) <br>
    

- This implies that : <br>
    - the distribution of $y^{i}$ given $x^{i}\,$ parameterized by  $\theta$ follows the Gaussian Distribution of average $\theta^{T}x$ and variance $\sigma^{2}$
    
    &emsp;&emsp;&emsp; $\normalsize p(y^{i}\, \| \,x^{i};\,\theta)\, \sim~ \,\frac{1}{\sqrt{2\pi}\sigma}exp(\frac{-(y^{i}\,-\theta^{T}x^{i})}{2\sigma^{2}})$
    
    
- The function $p(y^{i}\, \| \,x^{i};\,\theta)$ can be explicitly veiwed as the likelihood of $y$ for a varying $\theta$  <br>
    
    &emsp;&emsp;&emsp; $\normalsize L(\theta)\,=\,p(y^{i}\, \| \,x^{i};\,\theta)$
    
<br/>

## 2.1. Likelihood Function : $ L(\theta)$
<br/>

- as we've made an IID assumption, the likelihood for entire training set can be computed as the product of each probability of $y^{i}$. <br>

    <img src="https://user-images.githubusercontent.com/92680829/227084393-1c03057f-b858-41a7-a524-9081b2aad3c1.png" width="400">

- Given this likelihood function, our probelm turn into finding the sets of $\theta$ that maximizies the probabilistic distribution of $y$ given by the $x$
- As the function $L(\theta)$ contains exponential term, we can make it simpler by taking log to the function to make it linear and also turn the product into summed form.

    <img src="https://user-images.githubusercontent.com/92680829/227085579-0ba83582-5630-4f4f-a5a9-9f9579d34b9a.png" width="420"> <br>

- Hence, maximizing $\ell(\theta)$ actually becomes same as minimizing $\sum\limits_{i=1}^{m}(y^{i}\,-\,\theta^{T}x^{i})$, which is the error term we've seen before.
- To summarize, optimizing $\theta$ with least-square approach to error term ($\epsilon^{i}$) corresponds to finding $\theta$ that gives maximized likelihood distribution of $p(y^{i})$

<br/>

---

<br/>

# 3. Classification with Logistic Regression

<br/>

- Logistic regression is used for the binary classification in which y takes only two discrete values, 0 and 1. 
- LR models the probability that the $y^{i}$ takes on a particular value given the $x^{i} $ parameterized by $\theta$. 
- Logistic function, which maps the input values to a value between 0 and 1, representing the probability of $y^{i}$ taking the value 1.

- To map the input values ($x$) to proability with range [0, 1], we need to change the form of ``hypothese function using sigmoid function`` that converts the input values defined from negative to positive infinity into the output values from 0 to 1. <br>
    
    &emsp;&emsp;&emsp; $ \normalsize h_{\theta}(x) = g(\theta^{T}x) = \large \frac{1}{1+e^{-\theta^{T}x}} $ where $\normalsize \, g(z)\,=\,\frac{1}{1+e^{-z}}$

    <img src="https://user-images.githubusercontent.com/92680829/227090153-b7ee412d-1135-40b8-9520-d486cb20cf30.png" width="340"> <br>

    - $g(z)$ goes toward 1 as z goes to positive infinity and 0 as z goes to negative infinity, bounded by [0, 1]
  
<br/>
  
## 3.1. Maximum Likelihood Estimator

<br/>

- To fit the best estimate of $\theta$, we need to define the likelihood function for logistic classifier same as we did for linear regression.
- Probaility Function : get $(h_{\theta}(x))$ when y = 1 and get $1\,-\,(h_{\theta}(x))$ when y equals to 0 <br>

    &emsp;&emsp;&emsp; $ P(y\,=\,1\, \| x;\theta\,)\,=\,h_{\theta}(x) $ <br>
    &emsp;&emsp;&emsp; $ P(y\,=\,0\, \| x;\theta\,)\,=\,1\,-\,h_{\theta}(x)$ <br>
    &emsp;&emsp;&emsp; Both combined, $ P(y\,|\,x;\theta\,)\,=\,(h_{\theta}(x))^{y}\,(1\,-h_{\theta}(x))^{1-y} $ <br>
    

- Each data point is in IID, likelihood for entire dataset equals to product of the probability for each $y^{i}$ <br>
    <img src="https://user-images.githubusercontent.com/92680829/227101948-2e70c490-f08a-4be4-9c4f-87c1101ef88f.png" width="400"> <br>
    
    - ``Log Likelihood`` for easier Optimization : <br>
    
    &emsp;&emsp;&emsp; $\normalsize \ell(\theta)\,=\,logL(\theta) = \sum\limits_{i=1}^{m}\,y\,log(h_{\theta})\,+\,(1-y)\,log(1\,-\,h_{\theta}(x))$
     
<br/>
    
## 3.2. Maximization with Gradient Ascent

<br/>

&emsp;&emsp;&emsp; $\normalsize \theta_{j} := \theta_{j} \,\, + \,\, \alpha\frac{\partial \ell(\theta)}{\partial \theta_{j}}$ <br>

<img width="566" alt="Screen Shot 2023-03-23 at 9 48 42 PM" src="https://user-images.githubusercontent.com/92680829/227210477-1ded6090-2cc8-4e56-953c-1caaecd7c4d8.png"> <br>

- The explicit form of optimizing equation looks almost identical with gradient descent for linear regression, but the hypotheses function ($h_{\theta}(x)$) is different. 

<br/>

---

<br/>

# 4. Newton's Algorithm

<br/>

- Newton's algorithm, also known as Newton-Raphson method, is an iterative numerical method for finding the roots of a differentiable function (root : the point where $ f(x)\, = \,0$). 
- Finds the root of first derivative of log likelihood function ($\ell'(\theta)$) using sercond derivative.
<br>

- **Algorithm** <br>

    <img width="481" alt="Screen Shot 2023-03-23 at 9 48 47 PM" src="https://user-images.githubusercontent.com/92680829/227210836-e52b06ba-6d4a-48dc-bd3f-1560d8d609f1.png"> <br>
    
    1. set initial $\theta_{j}$ as random value and approximates next optimized $\theta_{j}$ by drawing a line tangent to the function at the currest guess of $\theta$ 
    2. solve for the point where that linear function equals to zero. 
    3. repeat 1. and 2. untll covergence of $\theta$
   

- Advantage of Newton's method is that it takes less computations needed to converge each $\theta$
- But the amount of computations grows with the number of parameters to fit. 


