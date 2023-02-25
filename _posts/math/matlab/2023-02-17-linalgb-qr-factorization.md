---
layout : post
title : "[MATLAB] Gram-Schmidt A = QR Factorization"
img : matlab/matlab.jpg
categories : 
    - [math-matlab]
tag : [Math, Linear Algebra, MATLAB, Factorization]
toc : true
toc_sticky : true
---

<br/>

## MATLAB code for A = QR Factorization
- Gram-Schmidt
- Creates orthogonal columns from independent set of columns of A.
- R is an upper triangular matrix with the length of each orthogonalized column of A on its main diagonal.

<br/>

### 1. Function for A = QR Factorization


```matlab
function [Q R] = qr_factorize(A)
% A = QR factorization (Gram-Schmidt)

[n, n] = size(A); 

for j = 1:n;
    v = A(:, j);     % pick a column to orthogonalize
    for i = 1:j-1;
        R(i, j) = Q(:, i)'*v;      
        v = v - Q(:, i)*R(i, j);    % orthogonalize (subtracts earlier projections)
    end
    R(j, j) = norm(v);      % set diagonal of A with the length of orthogonalized column of A
    Q(:, j) = v/R(j, j);
end s
```

<br/>

### 2. Example solved 


```matlab
A = [2 0 1; 2 -3 0; -1 3 0];

qr_factorize(A) 
ans =
 
    0.6667    0.6667    0.3333
    0.6667   -0.3333   -0.6667
   -0.3333    0.6667   -0.6667
```

<br/>


- Reference 
    - Strang, Gilbert. Introduction to Linear Algebra. 4th ed (Chapter 4. Orthogonality, p237)


