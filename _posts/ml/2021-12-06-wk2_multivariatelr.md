---
title : "[Coursera : ML Specialization] - Multivariate Linear Regression"
categories : 
    - Machine Learning
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

## **Multiple Features**

  <img src="https://user-images.githubusercontent.com/92680829/144761037-8ec93e76-35d2-4b0d-96b2-43e4021b1bd1.png" >

- suppose X0 = 1, you can simply cost function by using matrix
<img src="https://user-images.githubusercontent.com/92680829/144761081-afeaa0ee-6f0e-4b5b-846d-37333fd58808.png" >


## **Gradient Descent for Multiple Variables**
<img src="https://user-images.githubusercontent.com/92680829/144761158-69d7d496-dc31-45bb-966b-e2c6b52a1d18.png" >


### **Feature Scaling** to Speed Up Gradient Descent
- can speed up gradient descent by having each of our input values in roughly the same range. 
- θ will descend quickly on small ranges 
- On the otherhand, **θ will go down slowly on large ranges**
- and also will **oscillate inefficiently** down to the optimum when the variables are very uneven.

<img src="https://user-images.githubusercontent.com/92680829/144761265-7005db75-1afb-4004-af63-18f2ee9d4a85.png" >

- make sure all features are raning in approximately -1 ~ +1
- **Mean Normalization**
<img src="https://user-images.githubusercontent.com/92680829/144761403-11193870-cf01-4609-af02-d9afd71a0722.png" >

  - **μi** is the average of all the values for feature (i)
  - **Si** is either (max - min) or standard deviation (sd) of feature (i)

### **Debugging by adjusting Learning Rate (alpha)**
- Make a plot with number of iterations on the x-axis. Now plot the cost function, **J(θ) over the number of iterations of gradient descent**

<img src="https://user-images.githubusercontent.com/92680829/144761501-702c339b-02e3-4491-b9dd-1ab5066b8bbe.png" width="500px">

- Wrong case 

  <img src="https://user-images.githubusercontent.com/92680829/144761530-6c6a61c2-f332-4622-8a68-588cea9b05cf.png" width="600px">

  - adjust learning rate smaller.
  - with sufficiently small learning rate, gradient descent can always converge.
  - too small learning rate will lead to slow convergence

## **Features and Polynomial Regression**
- Our hypothesis function need not be linear (a straight line) if that does not fit the data well.
- change the behavior or curve of our hypothesis function by making it a **quadratic**, **cubic** or **square root** function and just make new features.

  - E.g)

    <img src="https://user-images.githubusercontent.com/92680829/144761725-9d710476-cce8-4c43-bd0f-f590b84c0ed8.png" >

  - x^2 = x(2), x^3 = x(3) 
  - polynomial regression can be converted to multivariate linear regression

- make sure you Do **Feature Scaling**

## **Normal Equation**
- gradient descent require multiple iterations untill it reahes to optimum, but **normal equations can compute equivalently optimal theta without those iterations**
- also no need to do feature scaling
- (X'X)-1X'y

<img src="https://user-images.githubusercontent.com/92680829/144763082-3c46cb9d-13ee-489f-82e3-e6b7ce8735d0.png" >

- Comparison between **Gradient Descen**t vs **Normal Equation**

<img src="https://user-images.githubusercontent.com/92680829/144763131-7470e0ff-7f5d-4cdf-b473-43e54e6a3595.png" >
