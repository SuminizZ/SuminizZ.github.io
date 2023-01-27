---
layout : post
title : "[Coursera : ML Specialization] - Debugging a Learning Algorithm"
date: 2022-04-10 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>

## **Debugging a learning algorithm**
<br/>

- There are many things you can do;
    1. Get more training data --> fix high variance (compensate overfitting)
        - you should always do some preliminary testing to make sure more data will actually make a difference (discussed later)
    2. Try a smaller set a features --> fix high variance
        You can do this by hand, or use some dimensionality reduction technique (e.g. PCA)
    3. Try getting additional features --> fix high bias 
    4. Adding polynomial features --> fix high bias 
        - Can be risky if you accidentally over fit your data by creating new features which are inherently specific/relevant to your training data
    5. Try decreasing or increasing λ --> fix either high bias or variance
        - apporpriate regularization

- There are some simple techniques which can let you rule out half the things on the list, which can help you save a lot of time!

<br/>

## **Machine Learning Diagnostic**
<br/>

- a test to gain insight what is/isn't working with your learning algorithm, and get guidance how best to improve its performance.

---
<br/>

## **How to Evaluate Hypothesis Function**
<br/>

- Evaluate the Generalization Power of your Hypothesis
    - model with low training error and high test error can be a sign of overfitting 
    - training error is not a good estimate for actual generalization error of your model
    - However, picking the model with smallest test error will end up in model that is overfitted to only test set, not the new examples that model has never seen before
    - Try, split the  dataset into three parts, **1. training set (60), 2. cross-validation set (cv) (20), 3. test set (20)**
    - We can now calculate three **separate error values for the three different sets using the following method**:
        1. Optimize the parameters in Θ using the training set for each polynomial degree.
        2. select the model with the least error using the cross validation set.
        3. Estimate the generalization error using the test set with 
        This way, the degree of the polynomial d has not been trained using the test set.
    - I think I can deal with it using **GridSearch** 

<br/>

## **Diagnosing Bias (underfitting) vs Variance (overfitting)**
<br/>

- <img src="https://user-images.githubusercontent.com/92680829/157142532-07133edf-8ca0-4b39-9b6b-6efff5dec38c.png" width="450" >

- Draw Error Plot using Training set and CV set 
    - <img src="https://user-images.githubusercontent.com/92680829/157142795-d1b88587-a280-4997-99e5-7aa2a2117a1d.png" width="500" >

<br/>         
                
- if d is too small --> this probably corresponds to a high bias problem
- if d is too large --> this probably corresponds to a high variance problem
- **For the high bias case (under-fitting)**
    - we find both cross validation and training error are high
- **For high variance (over-fitting)**
    - we find the cross validation error is high but training error is low
    - training set fits well, But generalizes poorly
- the point near where the Jcv value reach the minimum can be optimal 'd' 

<br/>

## **Regularization and Bias/Variance**
<br/>

- How we can automatically choose good regularization parameter λ
    - too large λ --> too simple model (high bias)
    - too small λ --> too fitted model (high variance)
    - Plot Jcv (w/o λ) and J train,  against each λ value 
    
    <img src="https://user-images.githubusercontent.com/92680829/157145429-e5cd04b7-cf93-4e9d-89a8-135c5b769671.png" width="600" >

<br/>

---

## **Learning Curves**
<br/>

- The more training data you have, 
    - the harder you could fit your hypothesis to all training data --> J train will increase
    - the better you can generalize your hypothesis to more new examples --> J cv will decrase
- So basically, leaning curves against training set size looks like below,

    - <img src="https://user-images.githubusercontent.com/92680829/157146471-1474ffea-f65e-4435-ae2b-a89e35dae3f2.png" wdith="350" > 

<br/>

### 1. In case of **High Bias**
<br/>

<img src="https://user-images.githubusercontent.com/92680829/157146854-76428e78-74fa-4a3d-8ae8-53b36af8ee9b.png" width="500" >

<br/>

- With the strong bias, as the hypothesis is too simple, so large training set doesn't really help to improve hypothesis
- Eventually, J train and J cv both will converge to the similarly high error 

<br/>

### 2. In case of **High Variance** (small λ)
<br/>

<img src="https://user-images.githubusercontent.com/92680829/157147772-677071d8-c7f8-435a-af28-ad57d8249ff5.png" width="500" >
 <br/>
   
- When the training size is relatively small, the error gap between J train and J cv is large due to the high variance
- However, even with the high variance, as training size gets larger, hypothesis can cover more and more new examples   so it will have better generalization power --> J cv will keep decrasing
- But still, large training size will make it more harder for the hypothesis to fit all the data --> J train will be saturated.
- Gap is getting smaller and smaller as training size gets bigger

---
<br/>

## **Neural Networks and Overfitting**
<br/>

- <img src="https://user-images.githubusercontent.com/92680829/157149592-7ab87882-b045-4599-acbf-ffd110dc506d.png" width="500" >

- when the model is suffering from high variance ( J cv >> J train) 
- --> it won't gonna help to add more hidden units or layers as it will end up increasing variance even more


