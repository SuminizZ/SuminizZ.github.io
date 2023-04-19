---
layout: post
title : "[Stanford CS229 03] Generalized Linear Model (GLM) and Softmax Regression"
img: ml/cs229.png
categories: [ml-cs229] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : truer
---
<br/>

## OUTLINES

1. Exponential Family
2. Generalized Linear Models
3. Softmax Regression (Multiclass Classification)

<br/>

# 1. Exponential Family

<br/>

- Distribution is said to belong to the exponential family if its probability density function (pdf) or probability mass function (pmf) can be expressed in the following form <br>

    &emsp;&emsp;&emsp;&emsp; $\large f(y\, \|\ x\,;\theta) = b(y)\,e^{(\eta^{T}T(y)\, - \,a(\eta))} $ <br>
    <br>
    
    - y : response variable 
    - $\eta$ : natural parameter (link function, $f(\theta)$)
    - $T(y)$ : sufficient statistics (function of y, mostly just T(y) = y)
    - $b(y)$ : Base measure
    - $a(\eta)$ : log partition number (normalizing parameter to make integral over entire domain be 1)
    
    <br>
    
- Sufficient Statistics $T(y)$
    - a function that holds the sufficient information of the data needed to estimate the parameters of interest in statistical model
    - if a sufficient statistic is available, then one can estimate the parameter of interest without using the full data
    - can be found in the exponential family
    - GLM uses only this sufficient statistics for an optimization process of parameters.

<br/>

    
## 1.1. Probability Distribution within Exponential Family 

<br/>

- The exponential family includes a wide range of commonly used probability distributions, such as the ``normal distribution``, ``Poisson distribution``, ``gamma distribution``, and ``binomial distribution``

- There are distinct data types matched for each probability distribution 
    - Gaussian : real numbers
    - Bernoulli : binary discrete numbers
    - Poisson : discrete, natural integer
    - Gamma or Exponential : postivie real numbers
    
<br/>

### 1.1.1. Gaussian Distribution 

<br/>

<img width="445" alt="Screen Shot 2023-03-23 at 9 48 57 PM" src="https://user-images.githubusercontent.com/92680829/227210157-40653363-5016-4c18-a789-15f2959b8710.png">

<br>

### 1.1.2. Bernoulli Distribution 

<br/>

<img width="485" alt="Screen Shot 2023-03-23 at 9 49 02 PM" src="https://user-images.githubusercontent.com/92680829/227210029-35fd18b7-0459-4b68-96f5-afae5a198de1.png">

<br>

### 1.1.3. Poisson Distribution 

<br/>

<img width="400" alt="Screen Shot 2023-03-23 at 9 49 05 PM" src="https://user-images.githubusercontent.com/92680829/227209938-278da525-56aa-4066-91b7-d8ef63d014e2.png">

<br/>


---

<br/>


# 2. Generalized Linear Models (GLMs)

<br/>

- Extends the linear regression model to handle data type not in normal distribution, such as binary or discrete count data
- To use GLMs, response variable (y) is assumed to be distributed in the form of exponential family 
- exponential family form have link function (or response function) that links the non-normal response variable y to linear predictors (x parameterized by $\theta$)
- The GLM can be trained using maximum likelihood estimation or Bayesian methods, and the parameters of the model can be estimated using numerical optimization algorithms.

<br/>


## 2.1. Maximum-Likelihood Function of GLMs

<br/>

### 2.1.1. Properties of GLM 

<br/>

- ``Convexity`` : MLE with respect to $\eta$ is concave function (or Negative log likelihood is convex) 
    -> guarantees convergence
    
- $E(T(y)) = \large \frac{\partial a(\eta)}{\partial \eta}$
- $V(T(y)) = \large \frac{\partial^{2} a(\eta)}{\partial \eta^{2}}$ -> positive definite

<br/>

### 2.1.2. Mean and Variance of Sufficient Statistics with Derivatives of $a(\eta)$

<br/>

- **$E(T(y))$**

    1. GLM is normalized with log partition number $a(\eta)$ so that its integral equals to 1.
    2. take derivative to the integral with respect to $\eta$ 
    3. can get the relation that $\,\,\, \large -\frac{\nabla g(\eta)}{g(\eta)}\, =\, \int T(y)g(\eta)b(y)e^{\eta^{T}T(y)}dy \,\, = E(T(y)) \,$ (here, $\large g(\eta)\, =\, e^{-a(\eta)} $) <br>
    <br/>

    <img width="619" alt="Screen Shot 2023-04-04 at 9 16 13 PM" src="https://user-images.githubusercontent.com/92680829/229789349-6d16d223-9bdc-4be1-9675-d37e75027be2.png">


- **$V(T(y))$**
    - take derivative to $E(T(y))$ with respect to $\eta$ to get $\large \frac{\partial^{2} a(\eta)}{\partial \eta^{2}}$ <br>
    <br/>
    
    <img width="854" alt="Screen Shot 2023-04-04 at 9 17 06 PM" src="https://user-images.githubusercontent.com/92680829/229790251-df6621eb-6a5c-41ce-af73-1b59abcf7229.png">

<br/>

### 2.1.3. Maximizing Log Likelihood of GLM 
    
<br/>

- take derivative to log likelihood with respect to $\eta$ and set it to be 0. (maximum point of concave function)

    <img width="761" alt="Screen Shot 2023-04-04 at 9 17 12 PM" src="https://user-images.githubusercontent.com/92680829/229790508-c5679881-6a1c-4c19-b30f-964565a117de.png">
    
    - solve the equation $\large \,\nabla a(\eta) = \frac{1}{N} \sum \limits_{i}^{N} T(y)$ gives you the natural parameter $\eta$ that maximizes the likelihood of GLM
    - Hence, you only need to keep the sufficient statistics term for learning process, instead of storing the full data.
    - as N (size of sample) goes to infinity, $\large \nabla a(\eta)$ reaches to $\large E(T(y))$
    
    
<br/>

### Design Choices for GLM in Machine Learning
    
<br/>

1. response variable (y) is from exponential family 
2. $\large \eta = \theta^{T}x$ 
3. output $\,\,\large E(y\, \|\ \,x;\theta) = h_{\theta}(x)$

<img width="757" alt="Screen Shot 2023-04-04 at 9 17 39 PM" src="https://user-images.githubusercontent.com/92680829/229790696-fc1061cf-517e-45fa-a6b0-2dc1a02d401a.png">

    
<br/>

---
    
<br/>

# 3. Softmax Regression (Multiclass Classification)
    
<br/>

- Known as ``Multinomial Logistic Regression``, is a supervised learning algorithm used for classification problems where the output variable is categorical with more than two possible outcomes
- Estimate the conditional probability distribution of the output variable (class) given the input variables
- Output variables $Y = \{y_{1}, y_{2}, ..., y_{k}, ... y_{N}\} $, each $y_{k}$ represents the probability that the given input $x$ belongs to the correspondig category k 
    - $\large \sum\limits_{k=1}^{N}\, y_{k}\, = \,1\,\,$  (N : number of categories)
       
<br/>
 
## 3.1. Softmax Function ($h_{\theta}(x)$)
    
<br/>

- Transforms a vector of real numbers (input variables) into a probability distribution (output) by ``exponentiating`` and ``normalizing`` the values <br>

    &emsp;&emsp;&emsp; $\large p(y^{i}_{k}\, \|\ x^{i} ; \theta)$ 
    
    &emsp;&emsp;&emsp;&emsp; = $\large \frac{e^{z^{i}}}{\sum\limits_{j=1}^{N} \, e^{z^{i}}}$ $(here,\; z = \theta^{T}x^{i})$ 

       
<br/>

## 3.2. Cost for softmax regression : Cross - Entropy
    
<br/>

- pretty much the same with the cost function (logistic cost) for binary classification <br>

    &emsp;&emsp;&emsp; $\large CE(\hat{y}, y) = -\sum\limits_{k=1}^{N}y_{k}log(\hat{y}_{k}) $
    
    - $\hat{y}^{i}_{k}\, $ : predicted probaility for category k
    - $y^{i}$ : real label (1 for correct category and 0 for others)
    
    
- penalizes when the probaility is low for the correct category
- encourages the model to assign high probabilities to the correct classes and low probabilities to the incorrect classes.  

