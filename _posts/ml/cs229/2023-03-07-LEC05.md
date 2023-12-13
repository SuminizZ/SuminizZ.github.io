---
layout: post
title : "[Stanford CS229 04] Generative Leaning - GDA & Naive Bayes"
img: ml/cs229.png
categories: [ml-cs229] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : truer
---
<br/>

## OUTLINES

- Generative Learning Algorithms
- GDA
- Naive Bayes 

<br/>

# 1. Generative Learning Algorithms

<br/>


- ``Generative Learning Algorithm``
    - model the underlying distribution of input features separately for each class (label, y)
    - first model the $p(y)$ and $p(x\, \|\ \,y)$ and use ``Bayes Rule`` to derive the posterior distribution of y given x
    - match new example to each model and find the class (y) that maximizes $p(y\, \|\ x)$
    - include Naive Bayes, Gaussian Mixture Models (GDA), and Hidden Markov Models

<br/>    

- ``Discriminative Learning Algorithm``
    - mapping the input features and output value $p(y\, \|\ \,x)$
    - directly predict the output based on input variables weighted by learned parameters
    - no need to know underlying distribution of input space

<br/>


---

<br/>


# 2. Gaussian Discriminative Analysis

- as one of the generative learning algorithms, this model makes an assumption that $p(x\, \|\ \,y)$ follows multivariate normal distribution 

<br/>


## 2.1. Multivariate Normal Distribution

<br/>


- $p(x  \|\  y)$ is parameterized by mean vector and Covariance matrix 

    - Mean vector : $\normalsize \mu\,\in\mathbb{R}^{n}$
    - Covariance matrix : $\normalsize \Sigma \in \mathbb{R}^{n \times n}$, where $\Sigma \geq 0$ is symmetric and positive definite <br>

    
    &emsp;&emsp;&emsp; $\normalsize p(x\, \|\ \,y)\, \sim \,N(\mu,\,\Sigma)$ <br>
    
    &emsp;&emsp;&emsp; $\normalsize p(x ; \,\mu, \,\Sigma) = \frac{1}{(2\pi)^{n/2}  \|\ \Sigma \|\ ^{1/2}}\,exp(-\frac{1}{2}\,(x - \mu)^{T}\,\Sigma^{-1}\,(x-\mu))$ <br>

    &emsp;&emsp;&emsp; $\normalsize E(x)\, =\, \mu$ <br>
    
    &emsp;&emsp;&emsp; $\normalsize \,Cov(X) = E((X\,-\,E(X))(X\,-\,E(X))^{T})\,=\, E(XX^{T}) - E(X)E(X)^{T}$ <br>


- ``Density of Multivariate Gaussian Distribution`` varies by $\Sigma$ and $\mu$ <br>

    - Diagnal entries of $\Sigma$ : determines the compression of pdf with respect to the direction parallel to each axis 
        - $\Sigma = I$ : standard normal distribution 
        - each represents pdf with $\Sigma$ equals to I , 2I, 0.4I, respectively 
        
       <img width="781" alt="Screen Shot 2023-03-29 at 9 10 27 PM" src="https://user-images.githubusercontent.com/92680829/228531295-aaf9f6cd-530f-4700-b19b-724aaecb10c5.png"> <br>
       
  - Off-diagonal entries (symmetric) : determines the compression towards the $45^{\circ}$ line between the axes of each feature
       - $\Sigma = \begin{bmatrix} 1\quad 0 \\ 0\quad 1 \end{bmatrix}$,  $\Sigma = \begin{bmatrix} 1\quad 0.5 \\ 0.5\quad 1 \end{bmatrix}$, $\Sigma = \begin{bmatrix} 1\quad 0.8 \\ 0.8\quad 1 \end{bmatrix}$ <br>
   
       <img width="781" src="https://user-images.githubusercontent.com/92680829/228532809-6ac82dc4-2840-4591-a8f0-042206faf67b.png">
       
  - varying $\mu$ moves the distribution along the axis 

<br/>

## 2.2. The Gaussian Discriminant Analysis (GDA) Model

<br/>


- classification problem in which input features $x$ are continuous random variables distributed in normal form and $y \in \{0, 1\}$ follows Bernoulli distribution

    <img width="220" alt="Screen Shot 2023-03-29 at 9 30 34 PM" src="https://user-images.githubusercontent.com/92680829/228535989-608a5252-8f7e-42e8-998e-1ac155dc3a96.png">
    

- tries to maximize the log-likelihood, which is the product of $p(x^{i}, y^{i} ; \phi, \mu_{0}, \mu_{1}, \Sigma)$
   
   &emsp;&emsp; $\normalsize \ell(\phi, \mu_{0}, \mu_{1}, \Sigma) = log\,\prod\, p(x^{i}, y^{i} ; \phi, \mu_{0}, \mu_{1}, \Sigma)$
   
   &emsp;&emsp; using Bayes Rule, can be expressed as <br>
   
   &emsp;&emsp; $\normalsize log\,\prod\, p(x^{i}\, \|\ \, y^{i} ; \phi, \mu_{0}, \mu_{1}, \Sigma)\,p(y^{i}\,;\,\phi)$
       
 
- Each distribution (class y=0 and y=1),

    <img width="600" alt="Screen Shot 2023-03-29 at 9 33 06 PM" src="https://user-images.githubusercontent.com/92680829/228540713-5ed10a55-78f1-4110-bf13-884a8d2fd5f1.png">


- the result of MLE : By maximizing the $\ell$ with respect to each paramter, find the best estimates of the parameters, 
    
    <img width="420" alt="Screen Shot 2023-03-29 at 9 52 59 PM" src="https://user-images.githubusercontent.com/92680829/228541298-5077b02f-52a2-4d6f-ac69-3501035cefcd.png">
    
- Predcit : Then, we can find the class of each training example that maximizes the log likelihood function 
    
    &emsp;&emsp; $\normalsize y^{i} = argmax{\,p(y^{i}\, \|\ \,x^{i})} = argmax(\,\large \frac{p(x^{i}  \|\  y^{i})\,p(y^{i})}{p(x^{i})})$
    
    &emsp;&emsp; $p(x^{i})$ is no more than a common constant for both classes, can ignore the demoninator.
    
    &emsp;&emsp; Hence, $\normalsize y^{i} = argmax(\,\large p(x^{i}  \|\  y^{i})\,p(y^{i}))$
    
  
- Pictorically, what the algorithm is actually doing can be seen in as follows, 
    
    <img width="472" alt="Screen Shot 2023-03-29 at 10 07 05 PM" src="https://user-images.githubusercontent.com/92680829/228544989-00d562a2-cb48-466f-af8f-eeea629014aa.png"> <br>
     
    - In summary, GDA models the distribution of input features $p(x  \|\  y=0)$ and $p(x  \|\  y=1)$ and calculate the $p(y^{i}  \|\  x^{i})$ as a product of $p(x^{i}  \|\  y^{i}) p(y^{i})$ using Bayes rule.
    - Then find the most likely output, maximizing the probability 


<br/>


## 2.3. GDA vs Logistic Regression

<br/>


- If we view the quantity $p(y=1 \,  \|\ \, x \,;\, \phi, \mu_{0}, \mu_{1}, \Sigma)$ as the function of $x$, we can find that it can actually be expressed in the following form,
    
    &emsp;&emsp; $p(y=1 \,  \|\ \, x \,;\, \phi, \mu_{0}, \mu_{1}, \Sigma)\,=\, \large \frac{1}{1\,+\,e^{-\theta^{T}x}}$ &emsp; , where $\theta$ is an appropriate function of $\phi, \mu_{0}, \mu_{1}, \Sigma$
    
    
- The converse, however, is not true. (logistic regression doesn't guarantee normally distributed x). <br> This means that GDA is stronger modeling assumption than logistic regression. <br> Hence, as long as the assumption is correct, GDA can make better prediction than logistic regression. 


- In contrast, logistic regression is less sensitive to incorrect modeling assumptions so that it's not significantly affected by the actual distrtibution of data (for example, Poisson distribution also makes $p(y \|\ x)$ logistic)


- To summarize, GDA can be more efficient and has better fit to the data when the modeling assumptions are at least approximately correct. <br> Logistic regression makes wearker assumptions, thus more robust to deviations from the modeling assumptions  

<br/>


---

<br/>


# 3. Naive Bayes 

<br/>


- Probabilistic classifiers based on applying Bayes' theorem with strong Naive Bayes (NB) assumptions between the features
- NB assumption assumes that each input feature is conditionally independent to each other given y (class), which is highly unlikely in reality.
- this algorithm still works okay even with this very "naive" assumption and provides clear advantage in terms of computational efficiency
- But for the data where input features are strongly correlated, the assumptions significantly limit its accuracy.

<br/>


## 3.1 Application of NB Algorithm as a Spam Classifier 

<br/>

- build a spam classifier that automatically classifies the email into spam or non-spam using Naive Bayes algorithm
- ``Training set`` : 
    - given an email with labeled with 1 for spam ($y^{i} = 1$) and 0 for non-spam ($y^{i} = 0$)
    - construct a feature vector whose lengith is equal to the number of words in vocab dictionary and each $jth$ feature represents whether $jth$ vocabulary is present in the mail $(x^{i}_{j} = 1)$ or not $(x^{i}_{j} = 0)$
    
    &emsp;&emsp; $ith$ email : $x^{i} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ .\\.\\.\\1\\0 \end{bmatrix}$
    
1. model $\normalsize p(x \|\ y)$ :
    - use NB assumption that features are conditionally independent within a class

    <img width="650" alt="Screen Shot 2023-04-01 at 1 39 30 PM" src="https://user-images.githubusercontent.com/92680829/229265955-d73e14f6-782a-443e-8883-894ec3de1eab.png">
    

2. Log-Likelihood function 
    
    &emsp;&emsp; $\normalsize L(\phi_{y}, \phi_{(j \|\ y=0)}, \phi_{(j \|\ y=1)}) = \prod_{i=1}^{m} \,p(x^{i}, y^{i})$ 
    
    &emsp;&emsp; $p(x^{i}, y^{i}) = \prod_{j=1}^{n}\,p(x^{i}_{j} \|\ y)\,p(y)$ &emsp;&emsp;, where each $p(x^{i}_{j} \|\ y)$ and $p(y)$ follows Bernoulli distribution 
     
3. MLE estimates 

    <img width="400" alt="Screen Shot 2023-04-01 at 1 48 44 PM" src="https://user-images.githubusercontent.com/92680829/229266215-df32267c-de7c-4b37-b1b5-16b949c56ae6.png">
    
    
4. Prediction 
    - find $argmax(y)\,\,p(y \|\ x)$
    
    <img width="700" alt="Screen Shot 2023-04-01 at 1 52 51 PM" src="https://user-images.githubusercontent.com/92680829/229266313-2b9683f4-d91e-4490-a3fe-45c6fe3e98eb.png">
    
    - repeat for y = 0, and select the class with max probability 
    
