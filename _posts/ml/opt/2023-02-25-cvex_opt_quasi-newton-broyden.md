---
layout: post
title : "[Convex Optimization] Quasi-Newton Method 1 : Broyden's Method"
img: ml/opt/convex.jpg
categories: [ml-opt] 
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : truer
---
<br/>

### Quasi-Newton method
- Iterative optimization algorithm used to solve unconstrained nonlinear optimization problems without explicit calculation for Jacobian or Hessian matrices. 
- Basic idea behind the Quasi-Newton method is to approximate the Hessian matrix of the objective function using an iterative formula that includes information from the gradient of the objective function and the difference between the current and previous iterates.

<br>

### Broyden's Method 

- to find the solution to a vector-valued function that represents the nonlinear system of equations that Broyden's method is used to solve. 
    - $f(x) \,= \,0$
    - for example, in logistic regression, the gradient of the likelihood function is used as the vector-valued function f(x).

- iterative update formula : $J_{k+1} = J_k + \frac{(f(x_{k+1}) - f(x_k) - J_k(x_{k+1} - x_k)) \cdot (x_{k+1} - x_k)^T}{\lVert x_{k+1} - x_k \rVert^2}$

- constructs an approximation to the Jacobian matrix 
- The update formula is based on the idea that the change in the Jacobian matrix is proportional to the change in the function values.

<br>

### Derivation

- [<span style="color:purple">**Convex Optimization 1 - Quasi-Newton Method : Broyden's Method**</span>](https://drive.google.com/file/d/1bg4O_4jck2AGwa9tO4XqU89ovHq4Kij5/view?usp=share_link){:target="_blank"}s

