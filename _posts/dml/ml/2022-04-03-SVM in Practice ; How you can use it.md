---
layout : post
title : "[Coursera : ML Specialization] - SVM in Practice"
date: 2022-04-05 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>


## **How to Use SVM with Kernels**
<br/>

<img src="https://user-images.githubusercontent.com/92680829/157623910-d62e41b6-0b4c-4c89-92e3-12b405c27ccb.png" width="600">
<br/>

- When to use linear kernel (No kernel)
    - when feature number (n) is large, while training set size (m) is small
- When to use Gaussian kernel (similarity)
    - when n is small, while m is large (ideally)
    - note **Do perform feature scailing before using Gaussian kernel function**
    
    <img src="https://user-images.githubusercontent.com/92680829/157625901-88bc1081-819d-46c1-8b65-4f63fe5e9b48.png" width="500">
    
<br/>

### **Restrictions for using Kernels : Mercer's Theorem**
<br/>

- Linear and Gaussian are most common, but not all similarity functions you develop are valid kernels
- Must satisfy Merecer's Theorem
- Other Types of Kernels (not that common..)
    - Polynomial Kernel
        - e.g ) (xT l)^2 , (xT l)^3 , (xT l+1)^3 , ...
        - General form is (xT l+Const)^Dim
        - If they're similar then the **inner product tends to be large**
        - Not used that often, Usually performs worse than the Gaussian kernel
            - Used when x and l are both non-negative
        - Two parameters
            - Degree of polynomial (D)
            - Number you add to l (Con)
    - String kernel
        - Used if input is text strings, Use for text classification
    - Chi-squared kernel
    - Histogram intersection kernel
    
<br/>

### **Multi-class classification for SVM**
<br/>

- Many packages have built in multi-class classification packages
- Otherwise use one-vs all method
- Not a big issue

<br/>

### **Logistic regression vs. SVM**
<br/>

- Logistic regression and SVM with a linear kernel are pretty similar

- if n (~10,000) is large (compared to m (10 ~1000)) --> use LR or SVM w/o kernel
- if n is small (1~10000) and m is intermediate( 10-50,000) --> use SVM with Gaussian Kernel
- if n is small and m is Large(50,000~) --> **Create more features**, then use LR or SVM w/o kernels

- It's not always clear how to chose an algorithm
    - Often more important to get enough data
    - Designing new features
    - Debugging the algorithm
