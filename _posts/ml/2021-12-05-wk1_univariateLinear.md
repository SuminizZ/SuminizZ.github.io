---
title : "[Coursera : ML Specialization] - Linear Regression with One Variable"
categories : 
    - Machine Learning
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

# **Linear Regression with One Variable**
- inear regression with one variable is also known as "univariate linear regression."
- predict a single output value y from a single input value x
- **Hypothesis Function** (일차방정식)

  <img src="https://user-images.githubusercontent.com/92680829/144719014-22384389-d016-4d46-a407-1ae01da4b75a.png">


## **Cost Function** : Hypothesis 의 예측성능 검증
  - **Mean Squarred Error (MSE)** : h(x) 함수값과 실제값(y) 의 차이의 제곱의 평균 
  <img src="https://user-images.githubusercontent.com/92680829/144719111-a1381f03-1631-49da-9c48-46daaa3c2dd6.png" width = "450px">
  
  - **Goal** : minimize J(θ0, θ1) with **Gradient Descent** 
  - Draw 3D contour figure to see how J changes with variations in θ0 and θ1
  <img src="https://user-images.githubusercontent.com/92680829/144719253-71d8c7cf-7d65-45e1-a20d-a1a0644a9a4a.png" width="550px"  >
  <br/>
  <img src="https://user-images.githubusercontent.com/92680829/144719312-59afd740-3df8-4e60-85d9-d91123572873.png" width="550px" >
  - the point where J minimizes is called **optima**, which represents the best hypothesis to predict the function between x and y and this can be gained from **Gradient Descent** method.

## **Gradient Descent**
<img src="https://user-images.githubusercontent.com/92680829/144719968-7396019b-7648-4dc4-b912-16602c910b7c.png" >

- point where red arrow is heading to is called local optimum where cost function become locally minimized.
- the way to find this point is to take partial derivative of theta 1 and theta 0 and update them using that derivative.

  <img src="https://user-images.githubusercontent.com/92680829/144720046-9c1cdba1-9380-4129-8a94-8004dddce617.png">
  
  <img src="https://user-images.githubusercontent.com/92680829/144721012-66efd7cb-1725-4541-8db5-4a393f889f0f.png" width="600px">

  - := this sign means to overwrite the left one with the right one. just '=' is no more than a truth assertion.
  - we need to **simultaneously** update theta 1 and theta 0, not one by one.
  - Updating a specific parameter prior to calculating another one on the j(th) iteration would yield to a wrong implementation.
  
  <img src="https://user-images.githubusercontent.com/92680829/144720115-ce31787c-a13a-4939-85b8-fcb8e2af75a1.png" >

  - **learning rate (lr)** : controls the rate of update.
    - too small : take so much time to reach the global opitmum
    - too large : can't converge, even more diverge
<img src="https://user-images.githubusercontent.com/92680829/144721040-7f402614-1b75-420f-8184-0a0bd5e32442.png" width="600px" >

    - gradient descent can converge to minimumn even with the **fixed learning rate**, as the derivative of theta (the magnitude of updating step) can automatically get smaller as it reaches to the optimum point
      
<br/>   
- **Gradient Descent Algorithm applied in Linear Regression** 
  - single example of derivation

    <img src="https://user-images.githubusercontent.com/92680829/144721465-536cc6ad-b77b-4ca5-bb23-82588d316acf.png" width="300px">
  
  - repeatedly apply derivation down below for theta 1 and theta 0 simultaneously untill there's no advance in J (convergence).
    <img src="https://user-images.githubusercontent.com/92680829/144721508-acd72abf-90ff-4f86-9ec6-e59b3460a6e2.png" width="500px" >

  - **Batch Gradient Descent** : use entire training sample for every each updating iterations.
  - **Convex Function** : for univariate linear regression, J is a convex function where gradient descent doesn't have any local optimum other than global optimum.

## **Linear Algebra Review**

