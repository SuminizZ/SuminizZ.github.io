---
layout : post
title : "[MATLAB] Solving Ax = b with Factorization A = LU"
categories : [math-linalgb]
tag : [Math, Linear Algebra, MATLAB, Factorization]
toc : true
toc_sticky : true
---

<br/>

## MATLAB code for solving Ax = b
- No Row Exchange
- A is invertible
- Use A = LU Factorization  

<br/>

### 1. Factorizes A to LU (No Row Exchange)

<br/>

```python
function [L, U] = factorize(A)
% square LU factorization with no row exchanges
[n, n] = size(A); 
zr = 1.e-6;

for k = 1:n
    if abs(A(k,k)) < zr
    end 
    L(k, k) = 1;   %set 1s on diagonal 
    for i = k+1:n
        L(i, k) = A(i, k)/A(k, k);   % computes multipliers
        for j = k+1:n
            A(i, j) = A(i, j) - L(i, k)*A(k, j);   % substracts mulitipliers times A(k) from A(i)
        end
    end
    for j = k:n
        U(k, j) = A(k, j);    % upper triangular matrix 
    end
end 
```

<br/>

### 2. Solve Lc = b & Ux = c to get solution x vector 
- Forward Elimination 
- Backward Substitution 

<br/>

```python
function x = slv(A, b)
% Solve Ax = b using Lc=b and Ux=c
[L, U] = factorize(A)
[n, n] = size(A);
s = 0;  
t = 0;

for k=1:n     % Forward Elimination Lc = b to slove c 
    for j=1:k-1
        s = s + L(k, j)*c(j);
    end
    c(k) = b(k) - s;
    s=0;
end

for k=n:-1:1     % Back-Substitution with Ux = c to solve from x(n) to x(1)
    for j=k+1:n     % from diagnoal to end of row
        t = t + U(k, j)*x(j);  % adds up U times previously earned x 
    end
    x(k) = (c(k) - t)/U(k, k);  % divide by pivot 
end
x = x'; 
```

<br/>


- Reference 
- Strang, Gilbert. Introduction to Linear Algebra. 4th ed (Chapter 2. Solving Linear Equation, p100)


