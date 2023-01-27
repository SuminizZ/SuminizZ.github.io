---
layout : post
title : "[Coursera : ML Specialization] - Anomaly Detection : Gaussian Distribution"
date: 2022-04-14 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>

## **Motivations** 
<br/>

- Anomaly detection (aka outlier analysis) is a step in data mining that identifies data points, events, and/or **observations that deviate from a dataset’s normal behavior**. 
- Anomalous data can indicate **critical incidents**, such as a technical glitch, or potential opportunities, for instance **a change in consumer behavior**
- Machine learning is progressively being used to automate anomaly detection.

    <img src="https://user-images.githubusercontent.com/92680829/158517578-d9d2cfcf-dadc-42ad-93d0-671ee3d2b75a.png" width="550" >

- ε is the probability that certain data point follows the usual distribution of data
- act as a threshold that distinguish whether certain data is anomalous or not.
- increasing ε can increase the number of anomalies and decreasing it can also decrease the data detected as an anomaly

<br/>

### **Use of Anomaly Detection**
- Fraud Detection
- Manufacturing
- Nonitoring computers in a data center

---

<br/>

## **Method : Gaussian (Normal) Distribution**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/158519573-9296ddc0-75d0-4211-bfb0-a6e91722e5d8.png" width="550" >

- probability density
- p(x; μ, σ2) : probability that x follows gaussian distribution parameterized by two elements μ(mean) and σ2 (variance)
- How distribution changes by those two parameters μ, σ2
    - <img src="https://user-images.githubusercontent.com/92680829/158519948-e7b86bcc-caf7-4231-b0e3-0284fbd7cf4b.png" width="500" >

<br/>

### **Parameters Estimation**
- how to calculate the parameters μ, σ2 with a given data
- not sure if it's real, it's just an approximation by given data
- but those estimated values become closer to real values as the data size gets bigger
- <img src="https://user-images.githubusercontent.com/92680829/158520435-5f91a354-6a29-460c-ae67-dfab8f504970.png" width="550" >


<br/>

### **Application : Anomaly Detection Algorithm**
- Model P(x) from the data set
- x is a vector (n-dimensional) 
- n : # of features / m : # of examples
- every each feature possesses each one's distinct parameter values, also, We model each of the features by assuming each feature is distributed according to a Gaussian distribution
- So model p(x) as 
    - **p(x1; μ1 , σ12) * p(x2; μ2 , σ22) * ... p(xn ; μn , σn2)**
    - in a strict way, it needs **independence assumption** for this equation to be true for describing P(x), but
    - practically, it just works fine even if each features are not independent from each other
- p(xi; μi , σi2)
    - The probability of feature xi given μi and σi2, using a Gaussian distribution

- **Density Estimation**

- <img src="https://user-images.githubusercontent.com/92680829/158522436-9a401797-0385-4ef6-8947-0711f1154eeb.png" width="550" >


- for j in 1 ~ n and for i in 1 ~ m
- we can estimate each features's parameters 
    - <img src="https://user-images.githubusercontent.com/92680829/158523055-510b41bf-5d74-4ed4-b0cb-052556ab9ffb.png" width="400" >


<br/>

#### **Algorithm Summary**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/158523215-baced271-ac36-4822-a4e8-53f892eeb81d.png" width="400" >

#### **Example : how to apply Gaussian Anomaly Detection**
<br/>

- <img src="https://user-images.githubusercontent.com/92680829/158523969-765fbc6f-92ee-43b6-bfe7-4c0a51413fb6.png" width="600" >

- P(x) = p(x(1), μ1 , σ12) * p(x(1), μ2 , σ22)
- any new example with p(x) smaller than ε are classified as 'anomaly'

---


<br/>

## **Developing and evaluating and anomaly detection system**

<br/>

### **Real-Number Evaluation System**

<br/>

- **Split Dataset**
    - we have some labeled data, of anomalous (y=1) and non-anomalous examples (y=0)
    - set training examples as a collection of normal examples (non-anomalous data)
    - prepare cv and test dataset mixed with anomalous and non-anomalous data
- **Algorithm evaluation**
    - Fit model **p(x)** by the prepared training set (all normal, y=0)
        - there's no label!!
    - On cross validation and test set, test the example x (should use different data in CV and test set)
        - y = 1 if p(x) < epsilon (anomalous)
        - y = 0 if p(x) >= epsilon (normal)
    - check the performance of model by using given labels
        - As we have Skewed data (almost normal), simple predict accuracy is not a good estimate of algorithm performance
        - Instead,
        - **Possible evaluation metrics**
            - Confusion matrix, Precision/Recall, F1-score
            - use **cv dataset** to decide most effective **ε**

<br/>

### **Anomaly detection vs. supervised learning**

<br/>

- If we have labeled data, why not use a supervised learning algorithm?
- When you should use supervised learning and when anomaly detection would be better

- **Anomaly Detection** : Fraud Detection, Manufacturing
    - Very small number of positive (y=1) examples
    - Large number of negative(y=0) examples ... can assume Gaussian distribution
    - There're many different types of anomalies
        - with very small number of positive examples, it's impossible for the model to learn every type of anomalies


- **Supervised Learning** : Spam emails, Cancer classification
    - When there are reasonably large positive and negative examples, large enough for the model to learn both

<br/>       

### **Choosing & Modifying features to use**

<br/>

- **Check if each feature follows Gaussian Distribution**
    - Non-Gaussian features
        - Use some valid **Transformation** to change the distribution more like Gaussian
            - log(x + c), sqaure root, x1/n..
            -
            - <img src="https://user-images.githubusercontent.com/92680829/158531096-b9eb07ab-2078-40cc-b82f-ed2378ad5502.png" width="500" >

<br/>

### **Debugging for Anomaly Detection**
-  Common problem
    - expectation : small p(x) for anomalous examples and large p(x) for normal examples
    - but, we often find p(x) for normal and anomalous examples is quite comparable
    - Like supervised learning error analysis procedure
        - Run algorithm on CV set
        - See which one it got wrong
        - Try come up with other new features to distinguish both based on trying to understand why the algorithm got those examples wrong

---

<br/>

## **Multivariate Gaussian Distribution**

<br/>

- Lets say in the test set we have an example which looks like an anomaly (e.g. x1 = 0.4, x2 = 1.5)
- Problem is, if we look at each feature individually they may fall within acceptable limits
    - <img src="https://user-images.githubusercontent.com/92680829/158533972-6963d40a-a8a0-423c-8ff7-91c9ca42b10c.png" width="550" >

<br/>

- However, even if we use both feature to distinguish the anomaly,
- With the previous Gaussian anomaly detection algorithm, that gree point will still be evaluated as normal data
    - <img src="https://user-images.githubusercontent.com/92680829/158534810-89ece414-5945-4d9a-be06-578e89363eac.png" width="550" >

- This is because our function makes probability prediction in concentric circles around the the means of both
- See two green and red x data points marked with red circle, it seems very clear to us that green point is an outlier while the red one is not,
- however, for our detection algorithm, both lied in a same radius of circle, so basically the same

- To get around this, we can use the **Multivariate Gaussian distribution**
    - don't model p(x1), p(x2), p(x3) ... p(xn) each and estimate the parameters μi and σi for each feature
    - instead, get P(x) from all features 1~n with parameters (μ, Σ), (Σ is covariance matrix)

    - <img src="https://user-images.githubusercontent.com/92680829/158536190-1f3d2453-0911-41e6-905a-dd1782646ea9.png" width="480" >


<br/>


###  **How the distribution of x1, x2 changes by Σ**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/158537463-0c71cbbc-288a-4506-9b43-c250dd8d6357.png" width="550" >

<br/>

#### **What if we modify the off-diagnal value of Σ : Capture Correlation between features**
- [i, j] element of matrix : correlation between xi and xj feature
    - not only the variance of the feature itself
    - it can reflect the covariance between different features (correlation)
- positivie and negative value each represents positive and negative correlation between those two values, respectively
- <img src="https://user-images.githubusercontent.com/92680829/158538453-c9f8d0bb-5162-4046-9ce4-c09061022956.png" width="400" >

<br/>

### **Applying multivariate Gaussian distribution to anomaly detection**

<br/>

- As mentioned, multivariate Gaussian modeling uses the following equation;
    - <img src="https://user-images.githubusercontent.com/92680829/158539253-8139c765-d825-4534-833d-a64f53543b3e.png" width="450" >

- Where 
    - μ - the mean of all n features (n-dimenisonal vector)
    - Σ - covariance matrix ([nxn] matrix)
        - automatically captures correlation between features
- If you have a set of examples {x1, x2, ..., xm }
    - The formula for estimating the parameters is suggested below,
    - <img src="https://user-images.githubusercontent.com/92680829/158539685-a659c5e9-830e-42b8-8f40-4c6c9c0d2f49.png" width="320" >

- **Detection Algorithm**
    1. fit model P(x) by setting μand Σ with given dataset
    2. Given a new example (test), compute P(x) by using equation below
        - <img src="https://user-images.githubusercontent.com/92680829/158539253-8139c765-d825-4534-833d-a64f53543b3e.png" width="450" >
    3. determine whether x is anomalous or not by comparing P(x) with epsilon value you set

- Actually the original Gaussian algorithm and the Multivariate Gaussian algorithm is quite similar
- except, original Gaussian model only has 0 values for all off-diagnal elements in Σ
    - actually, original model is just a special case of multivariate model
    - where there is an assumption that no correlation between any features at all

<br/>

#### **Comparison**
- <img src="https://user-images.githubusercontent.com/92680829/158542890-c7be143d-ee85-4d2d-a0ba-8dc6d7f20a50.png" width="500" >

