---
title : "[Coursera : ML Specialization] - Demensionality Reduction : Principal Analysis (PCA)" 
categories : 
    - Machine Learning
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

## **Motivations of Data Compression**
- Compression
    - Speeds up algorithms learning
    - Reduces memory and space used by data for them
    - Visualize your data 
- What is **dimensionality reduction**?
    - When you've collected unnecessarily many features
        - How to "simplify" your data set in a useful way
    - Example
        - Redundant data set - different units for same attribute
        - Reduce data to 1D (2D->1D)
    - <img src="https://user-images.githubusercontent.com/92680829/157796065-1fbf4fe7-8baf-45d2-8368-98552de76da8.png" width="250" >
      - x(i) : 2D --> z(i) : 1D
      

### **Reducing Dimensionality**
- if all the data lie on one plane, then we can reduce the 3D data to 2D
- project all data to a surface,
- specify the location along each axis on newly set plane 
- <img src="https://user-images.githubusercontent.com/92680829/157796675-4a6fce2e-8672-4307-bca6-b76046a91653.png" width="500" >

- x(i) : 3D --> z(i) : 2D
- if we reduce the dimension of training examples {x1, x2,... xm} where x(i) is n vector 
    - we can get lower dimension of {z1, z2...zm} where z(i) is k vector, and k <= n

### **Visualization of High-Dimensional Data**
- Dimensionality reduction can improve how we display information in a tractable manner for human consumption
- Example :
    - collect a large dataset about many facts of countries, let's say we have 50 features
    - it's very hard to visualize 50 Dimensional features
    - but if we reduce the Dimension from 50 to 2 features that can summarize those features, now we can easily plot the dataset by new features
- Typically you don't generally ascribe meaning to the new features (so we have to determine what these summary values mean)
    - e.g. may find horizontal axis corresponds to overall country size/economic activity
    - and y axis may be the per-person well being/economic activity
    
- So despite having 50 features, there may be two "dimensions" of information, with features associated with each of those dimensions
    
- **It's up to you to choose what of the features can be grouped to form summary features**, and how best to do that (**feature scaling** is probably important)    


---

## **Principal Component Analysis (PCA)**
- try to find a lower dimensional surface onto which data points can be projected
- try to find the surface that can minimize the squared distance between that surface and original data, which is called **"Projection Error"**
- **PCA tries to find the surface which has the minimum projection error**

- For 2D-1D PCA, we must find a vector u(1)
    - Onto which you can project the data so as to minimize the projection error
    - u(1) can be positive or negative (-u(1)) which makes no difference
    
    - <img src="https://user-images.githubusercontent.com/92680829/157801086-09ed948e-b78b-4602-9473-e3988072da17.png" width="300" >
    


- In more general cases, we may want to reduce from nD to kD surface
    - Find k vectors (u(1), u(2), ... u(k)) onto which to project the data to minimize the projection error
    - e.g. 3D->2D
        - Find pair of vectors which define a 2D plane (surface) onto which you're going to project your data
        - <img src="https://user-images.githubusercontent.com/92680829/157801877-0490fae6-b36e-4ed3-a4cb-1e7056f7685d.png" width="300" >

### **PCA and Linear Regression**
- you can find the cosmetic similarity between two of them, as both seems to calcuate the line(or surface) to minimize the sum of the squared distance between that line and original data
- However, they have clear difference in two way
    1. How to calculate the error
        - linear regression : it calculates the vertical distance (y difference) 
        - PCA : it calcuates the orthogonal distance (not y)
        - this gives very different effect 
    2. objective 
        - lr : this tries to predict "y" value by drawing the line that can best explain the relationship between x features
        - PCA : there is no "y", it just tries to reduce the dimension, nothing more than that

---

## **PCA Algorithm**
1. **Data Preprocessing !!**
    - feature scailing 
    - mean normalization : replace each xji with xji - μj (avg of feature j)
2. Compute the surface that minimizes the projection error
    - Compute the u vectors : The new planes
    - Compute the z vectors : z vectors are the new, lower dimensionality feature vectors    
    
- Then, how can we find the u vectors ?

### **Principal Component (PC) : Eigen Decomposition**
1. Compute **Covariance Matrix (Sigma)**
    - **[n x n] Square matrix giving the covariance between each pair of features**
    - shows how much the variance of x(i) and x(j) resembles each other
    
    <img src="https://user-images.githubusercontent.com/92680829/157807419-b21a3767-aaab-4575-ad01-c1bc44a54b9e.png" width="340" >
        
    - Divide Cov(x1,..xm) / m 
    - Note that x(i) is **"Mean Normalized"** (i)th example of [nx1] vector 
    
    - <img src="https://user-images.githubusercontent.com/92680829/157807701-9c052b37-3c56-452e-97a7-4ca8a1afa1aa.png"  width="300">

2. Compute **EigenVector** from Covariance Matrix of all features (you can use **svd function** too)
    - CV matrix have **n eigen vectors** in n-d matrix [nxn] : U matrix
    - Select the 1~k(th) longest eigen vectors that have maximal variance (maximal eigen value) to find k-d plane : Ureduce
        - principal surface [nxk] : plane that consists of the eigenvectors of a maximal variance from covariance matrix [nxn] 

3. Next we need to change x (n-d) to z (k-d) : Reduce Dimensionality
    - [Ureduce]T (kxn) x [x matrix] (nx1) = z matrix (kx1)
    - now we get the z matrix (k-dimensional) that are new features we got from PCA

---

## **Reconstruction from Compressed Data**
- from the compressed representation, you can go back to the approximation of your high-dimensional original representation
- <img src="https://user-images.githubusercontent.com/92680829/157820465-17d8d503-ac42-4045-a482-9878512b077f.png"  width="250">

- Considering z (vector) [kx1] = (Ureduce)T [kxn] * x [nx1], 
    - x(approx) [nx1] = Ureduce [nxk] x z [kx1]
    - <img src="https://user-images.githubusercontent.com/92680829/157821903-c1010a5a-764f-47c2-b9ec-6818336a0472.png" width="300" >

---

## **How to Use PCA in Practice**

### **Choosing the number of Principal Components**
- 1) PCA tries to minimize the projection error : sigma([[X(i) - Xapprox(i)]]^2)
    - this can actually be the **loss of original variance through the projection process**
- 2) PCA tries to maintain the original x variance as much as they can : sigma([[x(i)]]^2)
- Thus, make sure that you choose the k to be smallest value that can follow the below term
    - value within the range of 0.01~0.05 can be used
    - <img src="https://user-images.githubusercontent.com/92680829/157823110-39c12741-12e8-4b3b-8449-344697b24d7b.png" width="500" >

- It means that you can retain at least 99% of variance after applying PCA
- How to implement it ?
    - Plot, or prepare (k, variance) on validation set, and Select the k that gives the minimum acceptable variance, e.g. 90% or 99%.
    - [[Link] How to Implenment PCA in Scikit-Learn](https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0)
    - Seeing the plot below, you can see that k value over about 15 can retain great variability
    
    
    - <img src="https://user-images.githubusercontent.com/92680829/157825718-dc4e1e5e-3aab-4659-9c49-8baf33d63bf1.png" width="400" >

### **Supervised Learning Speed Up**
- when you have dataset with too much features (over 10,000)
- learning with all these features will severly slow down the training
- in this case, you can reduce the features by applying PCA

- Extract only x inputs from complete dataset (x(i), y(i)) 
- with earned unlabeled x dataset, you can apply PCA and get new unlabeled data with lower dimensionality z !
- Now you can get new training set (z(i), y(i))
- Make sure Mapping of x(i) to z(i) should be only defined by training examples !! 
- and then, you can apply this form of mapping to x(test) and x(cv)


### **Bad Use of PCA**
- to prevent overfitting 
    - PCA doesn't really care about the y value
    - Sol, it can ignore the useful information that y value could possibly give
    - Use regularization instead!
    
- Applying PCA first to desing ML system
    - Always use the original raw data first, and then observe the outcome of it 
    - Only if that doesn't work, you can try to include PCA step in your ML system design

---

### **-- Eigen Decomposition --** 
- Let's say **square matrix A as a linear transformation**, 
    - eigen vector : 
        - the non-zero vector that **gives itself multiplied by a constant value** from linear transformation A 
        - Matrix-vector multiplication --> Scalar-vector multiplication
        - the directions in which the data varies the most.
    - eigen value : that **constant value multiplied to eigen vector** after linear transformation
    - how to compute eigenvector and eigenvalue?
    
        - <img src="https://user-images.githubusercontent.com/92680829/157813105-0c5b0025-7854-4c60-9f37-ca2d14efa26e.png"  width="600">
        
        - Explain
            - Ax = λx 
            - we can set λ to λI, which makes no difference in overall equation
            - (A-λI)x = 0
            - as the eigenvector (x) is non-zero vector, (A-λI) vector cannot have inverse vector
            - therefore, det(A-λI) = 0
            - when you calculate the equation above, you can get the eigenvector x

