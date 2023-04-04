---
layout: post
title : "[Neural Networks and Deep Learning] Practice : Planar data classification with one hidden layer"
date: 2022-04-23 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---

<br/>

# **Planar data classification with one hidden layer**
- for this session, let's develoop a planar data classifier with shallow neural network with only 1 hidden layer 
- all references come from [here!](https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb)
- This chapter covers all below,
    - Implement a binary classification neural network with a single hidden layer
    - For binary classification, use activation function as non-linaer function such as tanh or sigmoid 
    - Compute the cross entropy loss
    - Implement forward and backward propagation to optimizes the weights 
    - Test Model Performance with Different Hidden Unit Size and Datasets

<br/>

## **1. Prepare required packages & Load Dataset**

```python
import unittest
!pip install testcase --trusted-host pypi.org --trusted-host files.pythonhosted.org
!pip install scikit-learn
```

- How to use **plt.contourf()** to draw contour plot
    - [Reference from here](https://m31phy.tistory.com/220)
    - [Official Documents here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
- [**np.c_**](https://rfriend.tistory.com/352)
    - <img src="https://user-images.githubusercontent.com/92680829/168471744-79a92162-1890-4a7c-9da7-bafe0ee4e2db.png"  width="400">


```python
# planar_utils.py

def plot_decision_boundary(model, X, Y, title):
    # Set min and max values and give it some padding
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    
    # Predict the function value for the whole grid
    Z = model(np.c_[x1x1.ravel(), x2x2.ravel()])
    Z = Z.reshape(x1x1.shape)
    
    # Plot the contour and training examples
    plt.contourf(x1x1, x2x2, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral, edgecolor='black')
    plt.title(title, fontsize=15)
    
def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    
    s = 1/(1+np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400         # number of examples
    N = int(m/2)       # number of points per class
    D = 2       # dimensionality (2 nodes in single hidden layer)
    X = np.zeros((m,D))       # data ma trix where each row is a single example 
    Y = np.zeros((m,1), dtype='uint8')       # labels vector (0 for red, 1 for blue)
    a = 4       # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2       # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2       # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

```


```python
import numpy as np 
import matplotlib.pyplot as plt
from testcase import *
import sklearn, sklearn.datasets, sklearn.linear_model

%matplotlib inline 

np.random.seed(1)  
```

- testCases : provides some test examples to assess the correctness of your functions
- np.random.seed(x) : numpy random seed is a numerical value that generates a **pseudo-random numbers**. The value in the numpy **random seed saves the state of randomness**. If we call the seed function using value 1 multiple times, the **computer displays the same random numbers**.


```python
## Load Dataset

X, Y = load_planar_dataset()
```


```python
print(X.shape)   # 2x400 matrix
print(Y.shape)   # 1x400 vector
print(Y.shape[1])   # training size

plt.figure(figsize=(8,6))
plt.scatter(X[0, :], X[1, :], c=Y, s=40, edgecolor='black', cmap=plt.cm.Spectral)     
plt.colorbar(label = 'color')     # red for 0, Blue for 1 
```
<img width="191" alt="image" src="https://user-images.githubusercontent.com/92680829/168481094-fa303545-673d-48a6-bc84-d031dbf0bf5c.png">

<img width="498" alt="image" src="https://user-images.githubusercontent.com/92680829/168481116-d3051cd2-0916-47ec-abb2-251b32e372d0.png">

<br/>

## **2. Simple Logistic Regression Classifier**
- before jumping right into developing classifier with shallow neural network, let's firstly make relatively simple classifier with logistic regression model
- Through this, you can compare the performances of those two different algorithms
- Use convenient sklearn packages to import lr classifer

```python
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
```

```python
lr_clf = sklearn.linear_model.LogisticRegressionCV()
lr_clf.fit(X.T, Y.T)    # X.T : (D, m)
```
<img width="240" alt="image" src="https://user-images.githubusercontent.com/92680829/168481147-88741a37-5701-4307-90c5-198bba4d9808.png">



```python
# plot decision boundary for separating classes (0, 1)

plot_decision_boundary(lambda x : lr_clf.predict(x), X, Y, "Logistic Regression")   
```

<img width="445" alt="image" src="https://user-images.githubusercontent.com/92680829/168481220-bdba2cd5-7655-4219-9b3e-eec03e270daa.png">



```python
# Prediction Accuracy 
from sklearn.metrics import accuracy_score

LR_result = lr_clf.predict(X.T)
print("Accuracy with Logistic Regression Classifier : {0}".format(accuracy_score(Y[0,:], LR_result)))


# instead of using sklearn library, you can also calcuate accuracy with code below
# dot result gives 1 only when Y and predicted value equals each other (either 0 or 1)

print(float((np.dot(Y, LR_result) + np.dot(1 - Y,1 - LR_result)) / float(Y.size)))   
```
<img width="519" alt="image" src="https://user-images.githubusercontent.com/92680829/168481244-07c9f91c-f327-4eff-8a55-0c61c62979b3.png">


- The performance of logistic regression algorithm represented as accuracy score was not great, which is 47%.
- This result implies that planar dataset is not linearly separable
- So you definitely need another algorithm, hope neural network with a single layer would work better 

<br/>

## **3. Define Neural Network Model Structure**
- Now, let's finally make Neural Network model with one hidden layer to predict classes for planar dataset
- Here is the representation of our model
    - <img src="https://user-images.githubusercontent.com/92680829/166148291-02e9da43-8f31-4841-bc11-e5a81683bbca.png" width="550" >

<br/>

### **Gradient Descent Loop**
- Implement forward propagation --> predict 
- Compute loss
- Implement backward propagation to get the gradients
- Update parameters (gradient descent)

<br/>

### **1) Forward Propagation** 
- 400 examples with two features, x1 and x2 
- Single hidden layer contains 4 training nodes with identical activation function **tanh**
- The activation function of output layer is sigmoid as it should return either 0 or 1 (threshold 0.5)

- <img src="https://user-images.githubusercontent.com/92680829/166148546-3dc96ddf-b90d-445d-94ac-7becc3f1c59b.png"  width="300">

<br/>

### **2) Computing Cost**

- Cost function of NN equals to that of logstic regression
- to make convex function

<img src="https://user-images.githubusercontent.com/92680829/166148666-3cd0a56f-764a-400c-ab74-788c673beb46.png"  width="450">

<br/>

### **3) Back-Propagation for Gradient Descent**
- J = J - α * dJ/dw (α, learning rate)
- For Gradient Descent, you need to calculate the partial derivative of Cost(L) by the parameter of interest 
- Then, if you want to compute **partial derivative of L** by **w (dw)**, then you firstly have to compute it by **a (da)**, then by **z (dz)**, and then finally by **w**, 
- Same applies to parameter **b**
    - **dL/dw (dW) = dL/da (da) * da/dz (dz) * dz/dw** 
    - How to back proagate with sigmoid activation function
    - <img src="https://user-images.githubusercontent.com/92680829/164571295-d51918ba-9d2f-4c60-add4-c6282b47bd85.png"  width="300">

- **Derivative of Multiple Activation Functions : da/dz (dz)**
    - here are the graphs and derivatives of various types of activation function
    - <img src="https://user-images.githubusercontent.com/92680829/167841964-d9483851-6903-45a3-8d4f-c96b399ceb0e.png"  width="600">

<br/>

### **Summary of Gradient Descent for Our Model**
- we have one **input layer** (400 examples with 2 features) : x (2, 400)
- **hidden layer** (4 nodes with 'tanh' activation function) : W[1] (4, 2), Z[1], b[1] --> a[1]
- one **output layer** (one node with sigmoid activation function) : W[2], Z[2], b[2] --> a[2]
    - <img src="https://user-images.githubusercontent.com/92680829/167842937-abd9c9b8-eb05-41d2-a16f-fe1935c20605.png" width="500" >
    
    - g[1]'(z[1]) here is 1-a[1]^2 (as g(z) is tanh(z))
    - 1 - np.pow(A1, 2)

<br/>

## **4. Build Model**
- We've just defined the model structure, decision boudary, cost function and the loop process for gradient descent!
- Let's build functions named 'nn_model()' to realize desired neural network

<br/>

#### Set sizes of each layer, input, hidden and output
```python

def layer_sizes(X, Y)s:
    nx = X.shape[0]    # (2, 400) -> 2 : size of input layer
    nh = 4             # nodes 9
    ny = Y.shape[0]    # (1, 400) -> 1 : size of output layer
    
    return (nx, nh, ny)
```
<br/>

#### Initialize model parameters randomly
```python
def initialize_params(nx, nh, ny):
    np.random.seed(2)     # set random state so that our outcomes have same value even with randomized initialization
    
    W1 = np.random.randn(nh, nx)*0.01 # from input to hidden layer (4, 2)
    b1 = np.zeros((nh, 1))     # (4, 1)
    W2 = np.random.randn(ny, nh)*0.01   # (1, 4)
    b2 = np.zeros((ny, 1))     # (1, 1)
    
    # confirm that each param has the right shape
    assert(W1.shape==(nh, nx) and b1.shape==(nh, 1) and W2.shape==(ny, nh) and b2.shape==(ny, 1))
    
    params = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
    
    return params
```
<br/>

#### Forward Propagation
```python
def fp(X, params):
    W1, b1, W2, b2 = (params['W1'], params['b1'], params['W2'], params['b2']) 
    
    Z1 = np.dot(W1, X) + b1  # (4, 2) x (2, 400) = (4, 400) + b(4, 1) broadcasting
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2  # (1, 4) x (4, 400) = (1, 400) + b(1, 1) broadcasting
    A2 = sigmoid(Z2)          # probability that x is 1
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {'Z1' : Z1, 'A1' : A1, 'Z2' : Z2, 'A2' : A2}
           
    return cache
```
<br/>

#### Calculate Cost
```python
# now calculate the cost (amount of deviation of A2 from Y)
# cost function (J(a))
    # J= −1/m (i=1~i=400∑( y(i)log(a[2](i)) + (1−y(i))log(1−a[2](i)) )

def compute_cost(A2, Y):
    m = Y.shape[1]
    
    tmp = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1-Y)
    cost = -np.sum(tmp)/m
    cost = float(np.squeeze(cost))    # remove axis whose size is 1 
    
    assert(isinstance(cost, float))
    return cost
```
<br/>

#### Compute gradient descent
```python
# now let's compute gradient descent of neural network with back-propagation

def bp(params, cache, X, Y):
    """"
    Returns: grads -- gradients with respect to different parameters (dW1, dW2, db1, db2)
    """
    m = X.shape[1]   # 400
    
    W1, b1, W2, b2 = (params['W1'], params['b1'], params['W2'], params['b2'])     
    Z1, A1, Z2, A2 = (cache['Z1'], cache['A1'], cache['Z2'], cache['A2'])
    
    dZ2 = A2 - Y       # (1, 400)
    dW2 = (1/m)*np.dot(dZ2, A1.T)      # (1, 400) x (400, 4) --> (1, 4)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)       # (1, 1)
    
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))   # (4, 1) x (1, 400) * (4, 400) --> (4, 400)
    dW1 = (1/m)*np.dot(dZ1, X.T)        # (4, 400) * (400, 2)  --> (4, 2)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)    # (1, 1) 
    
    grads = {'dW1': dW1, 'dW2': dW2, 'db1' : db1, 'db2': db2}
    
    return grads
```

<br/>

**Summary**
    - <img src="https://user-images.githubusercontent.com/92680829/169806728-ad382c6e-eac1-43b6-a7dc-2fa14d4c40d4.png"  width="550">

<br/>

#### Update Parameters
```python
def update_params(params, grads, lr = 1.2):
    W1, b1, W2, b2 = (params['W1'], params['b1'], params['W2'], params['b2'])     
    dW1, db1, dW2, db2 = (grads['dW1'], grads['db1'], grads['dW2'], grads['db2'])  
    
    W1 -= lr*dW1   # (4, 2)
    W2 -= lr*dW2    # (1, 4)
    b1 -= lr*db1    # (4, 1)
    b2 -= lr*db2    # (1, 1)
    
    updated_params = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
    
    return updated_params
```
<br/>

#### Build NN model
```python
def nn_model(X, Y, nh, num_iter, print_cost):
    """
    Arguments:
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by model (fp -> compute cost & bp -> update)
    """
    
    np.random.seed(3)
    nx, _, ny = layer_sizes(X, Y)
    
    params = initialize_params(nx, nh, ny)
    min_cost = float('inf')
    learning_curve = []
    
    for i in range(num_iter):    
        cache = fp(X, params)
        A2 = cache['A2']
        cost = compute_cost(A2, Y)
        grads = bp(params, cache, X, Y)
        params = update_params(params, grads)
        
        if cost < min_cost:
            min_cost = cost
            best_params = params
        
        if print_cost and i%10 == 0:
            learning_curve.append(cost)
            
            if i%1000 == 0:
                print("Cost after iterations {0} : {1}".format(i, cost))
                
    #   print(cost)
    
    return params, best_params, min_cost, learning_curve
    
```

- Now Finally, we've made our NN model!
- From now on, we will gonna predict the classes of examples (either 0 or 1) using our model
- Decision Rule
    - <img src="https://user-images.githubusercontent.com/92680829/168468017-a935af58-411e-437b-aef5-ea3adfef107e.png" width="500" >

<br/>

#### Predict classes of X
```python
def predict(best_params, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    cache = fp(X, best_params)
    A2 = cache['A2']    # predicted values  (not classes)
    
    preds = A2 > 0.5   # (1, 400)
    
    return preds
```


```python
# Result 
params, best_params, min_cost, learning_curve = nn_model(X, Y, 4, 10000, 1)
# plt.figure(figsize=(8, 5))
# plt.axis([0, 1000, 0, max(learning_curve)])        # [xmin, xmax, ymin, ymax]
plt.plot(learning_curve)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Learning Curve (Cost by Iterations)", size=15)
```
<br/>

<img width="460" alt="image" src="https://user-images.githubusercontent.com/92680829/168481446-71a848cf-cc9b-4f58-94d8-0d3f28c09afa.png">

<img width="525" alt="image" src="https://user-images.githubusercontent.com/92680829/168481497-1428fc0e-5016-48fc-a695-8d78f7ce9442.png">

<br/>

```python
# x : comes with shape (400, 2)
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x : predict(best_params, x.T), X, Y, "Neural Network Model with a Single Hidden Layer")
```
<img width="580" alt="image" src="https://user-images.githubusercontent.com/92680829/168481516-55e35ec9-5cf6-4aac-b0ca-ca2f30266ecb.png">


```python
def accuracy_score(preds, Y, nh):
    m = Y.shape[1]
    error = float((np.dot(1-Y, preds.T) + np.dot(Y, 1-preds.T))/m)   
    # same as np.sum(np.multiply(1-Y, preds) + np.multiply(Y, 1-preds))/m
    print("Accuracy with Neural Network with 1 Hidden Layer with {0} Units: {1} %".format(nh, (1-error)*100))
    
    return (1-error)*100
```


```python
_, best_params, _, _ = nn_model(X, Y, 4, 10000, 0)
preds = predict(best_params, X)
m = X.shape[1]

accuracy_score(preds, Y, 4)
```
<br/>

<img width="341" alt="image" src="https://user-images.githubusercontent.com/92680829/168481532-09476a66-87ac-47af-95f1-91151179b55f.png">

- From our NN model (1 hidden layer with 4 units), we've just gained 90% accuracy
- Previously, accuracy from Logistic Regression Classifier was 47%
- Now you can see NN with only one hidden layer can outperform Logistic regression
- Our NN model has learnt the leaf patterns of the flower, which shows NN can learn even highly non-linear decision boundaries, unlike logistic regression.

<br/>

## **5. Compare Accuracy of NN with Different Unit Sizes of Hidden Layer**

```python
plt.figure(figsize=(12, 24))
h_sizes = [1, 4, 8, 12, 16, 20, 50]

for i, nh in enumerate(h_sizes):
    plt.subplot(4, 2, i+1)
    _, best_params, _, _ = nn_model(X, Y, nh, 10000, 0)
    preds = predict(best_params, X)
    plot_decision_boundary(lambda x : predict(best_params, x.T), X, Y, "Hidden Units {0}".format(nh))
    accuracy_score(preds, Y, nh)
```
<br/>
<img width="647" alt="image" src="https://user-images.githubusercontent.com/92680829/168481546-e2b19f97-afe7-4dea-bad2-e79c3430b2ca.png">
    
<img width="630" alt="image" src="https://user-images.githubusercontent.com/92680829/168481570-724bce8a-e398-40e7-aab0-0bbf913bbe5b.png">
    
<br/>

### **Interpretations :**
- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.
- The best hidden layer size seems to be around nh 8. 
- Indeed, values greater than 8 seem to incur noticable overfitting as shown in decision boundary contour plot.
- You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting.

<br/>

## **6. Performance on Other Datasets**
- Now, let's test our model performance on other 4 datasets
- Unit size of 1 Hidden layer will be fixed as 8, which was figured as best unit size that prevents overfitting


```python
datasets = dict()
datasets['noisy_circles'], datasets['noisy_moons'], datasets['blobs'], datasets['gaussian_quantiles'], _ = load_extra_datasets()
```


```python
def accuracy_score_2(preds, Y, d):
    m = Y.shape[1]
    error = float((np.dot(1-Y, preds.T) + np.dot(Y, 1-preds.T))/m)   
    # same as np.sum(np.multiply(1-Y, preds) + np.multiply(Y, 1-preds))/m
    print("Accuracy for Dataset <{0}> with Unit Size 8 : {1}%".format(d, (1-error)*100))
    
    return (1-error)*100
```


```python
plt.figure(figsize=(12, 24))

for i, d in enumerate(datasets.keys()):
    plt.subplot(3, 2, i+1)
    X, Y = datasets[d]
    # print(X.shape, Y.shape)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    if d == 'blobs':
        Y = Y%2
    
    _, best_params, _, _ = nn_model(X, Y, 8, 10000, 0)
    preds = predict(best_params, X)
    plot_decision_boundary(lambda x : predict(best_params, x.T), X, Y, "Datasets : {0} with Unit size 8".format(d))
    accuracy_score_2(preds, Y, d)
```
<img width="617" alt="image" src="https://user-images.githubusercontent.com/92680829/168481747-301e12ff-4e7d-4d1f-86b0-503ee1a4b2ec.png">

<img width="617" alt="image" src="https://user-images.githubusercontent.com/92680829/168481765-a62b3c56-f302-4f68-8996-c5798073a4a8.png">

- Performance of our NN model quite differs by the datasets
- But, definitely can tell that our model can learn highly non-linear, complex decision boundaries with pretty fine accuracy
