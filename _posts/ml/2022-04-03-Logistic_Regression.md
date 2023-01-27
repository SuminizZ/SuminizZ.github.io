---
layout : post
title : "[Coursera : ML Specialization] - Logistic Regression"
date: 2022-04-03 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-ml] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>


## **Logistic Regression**
<br/>


- Classification Algorithmn
- Where y is a discrete value : Develop the logistic regression algorithm to determine what class a new input should fall into
- How do we develop a classification algorithm?
    Tumour size vs malignancy (0 or 1)
    - We could use linear regression
    - Then threshold the classifier output (i.e. anything over some value is yes, else no)
    - In our example below linear regression with thresholding seems to work
    - <img src="https://user-images.githubusercontent.com/92680829/156676623-16341baa-323a-4736-9d40-921fa77560c7.png" width="500" >
   
<br/>


### **Function that determines discrete classification**
<br/>


- We want our classifier to output values between 0 and 1
- When using linear regression we did hθ(x) = (θT x) = z
- For classification hypothesis representation we do hθ(x) = g((θT x)), where g(z) with z as a real number and g(z) is the final classified outcome, which is either 1 or 0.
- g(z) = 1/(1 + e-z)
- g(z) is the sigmoid function, or the logistic function
- when x infinitely increases, g(z) converge to 1, whereas x infinitely decreases, g(z) converge to 0.
- all g(z) values are within between 0 ~ 1
- <img src="https://user-images.githubusercontent.com/92680829/156677845-3711a946-4037-4051-a5d3-0fd47545acd7.png" width="400">


- Since this is a binary classification task we know y = 0 or 1
- P(y=1\|x ; θ) ... probability of y equal to 1, given x parameterized by θ
- So the following must be true 
    - P(y=1\|x ; θ) + P(y=0\|x ; θ) = 1 
    - P(y=0\|x ; θ) = 1 - P(y=1\| ; θ)

<br/>

### **Decision Boundary**
<br/>
  - x criteria that determines whether y is either 0 or 1
  - suppose that y =  1 when  hθ(x) >= 0.5, which means that z, θT x >= 0 and y = 0 when hθ(x) < 0.5, θT x < 0
  - for example, hθ(x) = g(θ0 + θ1x1 + θ2x2), where θ0 = -3, θ1 = 1, θ2 = 1
  - So, θT is a row vector = [-3,1,1]
  - The z here becomes θT x
  - We predict "y = 1" if -3x0 + 1x1 + 1x2 >= 0, so as x0 equals to 1, -3 + x1 + x2 >= 0
  - If (x1 + x2 >= 3) then we predict y = 1
  - Let's plot how z differs by each combination of x1, x2
  - the line x1 + x2 = 3, that separates g(z), y is the decision boundary
  
  - <img src="https://user-images.githubusercontent.com/92680829/156680691-a5498da1-0e86-4281-9135-5735a5f25b10.png" width="300">

- Non-Linear Decision Boundary
    - hθ(x) = g(θ0 + θ1x1+ θ3x12 + θ4x22)
    - Say θT was [-1,0,0,1,1] then we say;
    - Predict that "y = 1" if -1 + x12 + x22 >= 0 or x1^2 + x2^2 >= 1
    - If we plot this, This gives us a circle with a radius of 1 around 0
    - 
    <img src="https://user-images.githubusercontent.com/92680829/156681451-336e69b6-1fff-423f-8ad8-5bb580e91bb3.png" width="300">




---
<br/>


## **Cost Function of Logistic Regression**
<br/>


Training set of m training examples, Each example has is n+1 length column vector
x0 = 1
y ∈ {0,1}
Hypothesis is based on parameters (θ)

<img src="https://user-images.githubusercontent.com/92680829/156683168-6dfb6801-f65a-4a2c-815a-4d37f69839a8.png" width="500">
<br/>


- Given the training set how to we chose/fit θ?
    - Cost function of linear regression was like below, 
    - <img src="https://user-images.githubusercontent.com/92680829/156683283-033fa772-f636-4bdf-87f3-643d477483a1.png" width="300">
<br/>


- Instead of writing the squared error term, we can write 
- **cost(hθ(xi), y) = 1/2(hθ(xi) - yi)2**
- Which evaluates to the cost for an individual example using the same measure as used in linear regression
    - We can redefine J(θ) as
    - <img src="https://user-images.githubusercontent.com/92680829/156683371-e8fb4778-a11f-4199-a99e-3ca4de1588fa.png" width="300">

Which, appropriately, is the sum of all the individual costs over the training data (i.e. the same as linear regression)

- This is the cost you want the learning algorithm to pay if the outcome is hθ(x) and the actual outcome is y
- If we use this function for logistic regression, this is a non-convex function for parameter optimization
    - non-convex function : wavy - has some 'valleys' (local minima) that aren't as deep as the overall deepest 'valley' (global minimum).
    - Optimization algorithms can get stuck in the local minimum, and it can be hard to tell when this happens.
 - **A convex logistic regression cost function**
     - To get around this we need a different, convex Cost() function which means we can apply gradient descent
     - <img src="https://user-images.githubusercontent.com/92680829/156684011-edda5943-64ce-43b9-924b-7a7fd1ce0ddc.png" width="500">
<br/>


- This is our logistic regression cost function
    - This is the penalty the algorithm pays
    - Plot the function
- 1. Plot y = 1
    - So hθ(x) evaluates as -log(hθ(x))
    - <img src="https://user-images.githubusercontent.com/92680829/156685913-f0e750ef-56db-4deb-9a0d-f2cbcab3e3f5.png" width="270">
    
<br/>


2. plot y=0
    - So hθ(x) evaluates as -log(1-hθ(x))
    -
    - <img src="https://user-images.githubusercontent.com/92680829/156686219-e35d4c6c-2001-480a-9acd-cd927f906fb3.png" width="250">
    


<br/>


### **Simplified Cost Function and Gradient Descent**
<br/>


- Instead of separating cost function into two parts differing by the value of y (0 or 1),
- we can compress it into one cost function, which makes it more convenient to write out the cost.

    - **cost(hθ, (x),y) = -ylog( hθ(x) ) - (1-y)log( 1- hθ(x) )**
    - y can only be either 0 or 1
    - when y = 0, only -log( 1- hθ(x) ) part remains, which is exactly the same as the original one
    - when y =1, only -log( hθ(x) ) part remains
    -
    - <img src="https://user-images.githubusercontent.com/92680829/156687790-4532412e-706c-435c-b5aa-7d4a5f9145c3.png" width="600" >

<br/>


- Repeat Below to improve cost 
    - Interestingly, derivative of J(θ) of logistic regression is exactly identical with that of linear regression
    - 
    - <img src="https://user-images.githubusercontent.com/92680829/156696635-ab555f91-5544-40e9-9855-fe92787b3901.png" width="400">
<br/>



### **Derivative of Cost Function of Logistic Regression**
<br/>


- Step1 : get derivative of h(θ) = 1/(1 + e-z) 
    - <img src="https://user-images.githubusercontent.com/92680829/156696402-799da3b1-8d66-4ab4-b7e6-8e92c27d46f3.png" width="400">
<br/>

- Step2 : apply derivative to J(θ)

    - <img src="https://user-images.githubusercontent.com/92680829/156696529-a943aceb-f987-4324-9a57-b2ad41e3a35f.png" width="400">

- 
    - <img src="https://user-images.githubusercontent.com/92680829/156696553-9fbfbcfe-76a5-4e34-9fac-27300e9c4994.png" width="400">


- 
    - <img src="https://user-images.githubusercontent.com/92680829/156696592-9857ffd5-6637-46ed-abef-2f47d21b64c0.png" width="500">

<br/>


## **Advanced Optimization**
<br/>


Alternatively, instead of **gradient descent** to minimize the cost function we could use
- Conjugate gradient
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS (Limited memory - BFGS)
- Advantages
    - No need to manually pick alpha (learning rate)
    - Have a clever inner loop (line search algorithm) which tries a bunch of alpha values and picks a good one
    - Often faster than gradient descent
    - Do more than just pick a good learning rate
    - Can be used successfully without understanding their complexity
- Disadvantages
    - Could make debugging more difficult
    - Should not be implemented themselves
    - Different libraries may use different implementations - may hit performance


<br/>


##  **Multiclass Classification**
<br/>


- **One vs. all classification**
    - Split the training set into three separate binary classification problems
    - i.e. create a new fake training set
        - Triangle (1) vs crosses and squares (0) hθ1(x)
            - P(y=1 \| x1; θ)
        - Crosses (1) vs triangle and square (0) hθ2(x)
            - P(y=1 \| x2; θ)
        - Square (1) vs crosses and square (0) hθ3(x)
            - P(y=1 \| x3; θ)
            
- Overall
    - Train a logistic regression classifier hθ(i)(x) for each class i to predict the probability that y = i
    - On a new input, to make a prediction, 
        - **run all three classifiers on the input, and then pick the class i that maximizes the probability that hθ(i)(x) = 1** 

    <img src="https://user-images.githubusercontent.com/92680829/156708103-9d51b0cf-c811-4060-83b4-f3e383e04b96.png" width="250">
