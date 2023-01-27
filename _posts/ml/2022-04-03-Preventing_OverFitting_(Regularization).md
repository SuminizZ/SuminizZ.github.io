---
layout : post
title : "[Coursera : ML Specialization] - Regularization : Preventing Overfitting"
date: 2022-04-05 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>


## **Poor or Appropriate Fitting**
<br/>

- Data Fitting in Linear Regression
 - UnderFitting (Fit a linear function)
     - high bias : there's a strong preperception that data would follow the specific linear correlation
     - generalize too much
     - poor at predicting both training dataset and new data
 - Apporpriate Fitting (Fit a quadratic function)
     - appopriate generalization
 - OverFitting (Fit a 4th order polynomial)
     - high polynomial function used --> weak generalization
     - good at predicting training set, but not good at new untrained data
     - too many features with small data lead to overfitting
 
<br/>

 - <img src="https://user-images.githubusercontent.com/92680829/156711914-22c2922c-7146-453b-bc0c-e72ddd79a271.png" width="600" >
 
<br/>

- Data Fitting in Logistic Classification

   - <img src="https://user-images.githubusercontent.com/92680829/156712079-6671a1c0-bbe8-4407-8e9b-a0c63d4bad83.png" width="600" >


<br/>


## **Adress Overfitting**
<br/>

1. Reduce the number of features 
    - Manually select which features to keep and which to not
    - Model Selection Algorithm
    - problems : there're possibilities that we might loose valuable data that can actually contirbutes to improve predicting power of model
2. Regularization
    - Keep all the features, but reduce the magnitude/values of parameters θj, thus reducing the amount of effect that each variable can exert
    - penalize the features of less relevant to y and limit their contribution in predicting 
    - small values for parameters leads to "simpler" hypothesis and make the model less prone to overfitting
    - Then, how we can decide which parameters we should shrink in advance?
        - 1) **Add a term at the end**
            - This regularization term shrinks every parameter
            - By convention you don't penalize θ0 - minimization is from θ1 onwards
            - **λ is the regularization parameter** 
                - (1) Want to fit the training set well
                - (2) Want to keep parameters small
            -  <img src="https://user-images.githubusercontent.com/92680829/156714814-49b0a6f6-0aef-4b23-8849-1b7cb579d604.png" width="500" >
            <br/>

            - What if λ too big up to like 1e100 ? ... algorithm results in underfitting 
                - y alomost equals to θ0
                - all other parameters almost equal to 0 compared to λ
            - what if λ too small? ... can't achieve the original goal, that is to limit the effects of each parameter
            - <img src="https://user-images.githubusercontent.com/92680829/156717727-8a91b545-5f37-475d-a377-ddf745806908.png" width="400" >
           
                
<br/>


## **Regularized Linear Regression**
<br/>

- Previous Gradient Descent
    - <img src="https://user-images.githubusercontent.com/92680829/156717937-38fa5fe1-2a99-4b82-8345-29757925bb84.png" width="400">
        
- Regularized Gradient Descent 
    - <img src="https://user-images.githubusercontent.com/92680829/156718155-014e2729-3a6a-4c24-9f30-5880cd113574.png" width="500" >


- the term (1-α(λ/m))
    - necessarily smaller than 1 --> penalize each feature θj by the magnitude of λ
    - being multipled for every repeat, being penalized repeatedly
    
<br/>

## **Regularization by Normal Equation**
 - <img src="https://user-images.githubusercontent.com/92680829/156720941-c0b195ca-e3bf-41b7-8d1f-f10f5802fdff.png" width="500">

<br/>


## **Regularized Logistic Regression**
<br/>

- quite similar with regularized linear regression
- but, obviously the hypothesis is very different
- <img src="https://user-images.githubusercontent.com/92680829/156721480-d112d1d2-f5fa-4f2d-81bf-6ff2dcf07d7a.png" width="450"> 

