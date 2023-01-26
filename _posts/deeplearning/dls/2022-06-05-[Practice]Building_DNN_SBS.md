---
layout: post
title : "[Neural Networks and Deep Learning] Practice : Building DNN Step by Step"
date: 2022-06-05 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---


<br/>

# **Building Deep Neural Network : Step by Step**
- I've previoulsy made shallow planar classifier (with 1 hidden layer). For this week, I will build a real "Deep" neural network with as many layers as we want!
- This practice covers all below,
    - Use ReLu for all layers except the output layer with sigmoid activation function
    - Build multiple hidden layers (at least more than 1)
    - Implement easy-to-use neural network class


## **0. Outline of Practice**
- To build our neural network, we will define several "helper functions", which will be used later for building **2-layer neural network** and **L-layer neural network**
- Types of **helper functions** that will be defined
    - Intialize Parameters 
    - Forward Propagation (linear, Relu)
    - Compute Cost
    - Backward Propagation (linear, Relu)
    - Update Parameter (Gradient Descent)
- Summary of model
    - As an activation function, **Relu** for hidden layers (L-1 layers) and **Sigmoid** for output layer
    <img src="https://user-images.githubusercontent.com/92680829/172620970-7aaf7dbf-5e30-4d27-985d-5f64fb05bec0.png" width="620">



## **1. Load Packages** 
- TestCases : test cases to assess the correctness of your functions, got this from [<span style="color:blue">**here**</span>](https://github.com/knazeri/coursera/blob/master/deep-learning/1-neural-networks-and-deep-learning/4-building-your-deep-neural-network-step-by-step/testCases_v2.py){:target="_blank"}
- Activation Function (ReLu, Sigmoid) and its Derivative by Z (for Back-Propagation)


```python
import numpy as np
import h5py
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)      # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
 
%load_ext autoreload
%autoreload 2
 
np.random.seed(1)
```

- TestCases

```python
np.random.seed(1)

def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
                  [-1.62328545, 0.64667545],
                  [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    
    return A, W, b

def linear_activation_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
                  [-1.62328545, 0.64667545],
                  [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
                  [-1.62328545, 0.64667545],
                  [-1.74314104, -0.59664964]])
    parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
                                  [-1.07296862,  0.86540763, -2.3015387 ]]),
                  'W2': np.array([[ 1.74481176, -0.7612069 ]]),
                  'b1': np.array([[ 0.],
                                  [ 0.]]),
                  'b2': np.array([[ 0.]])}
    """
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])
    
    return Y, aL

def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(2,2)
    A = np.random.randn(3,2)
    W = np.random.randn(2,3)
    b = np.random.randn(2,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_activation_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    
    return dA, linear_activation_cache

def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ( (A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches

def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return parameters, grads
```
- Activation Function

```python
def sigmoid(Z):
    """
    Implement sigmoid activation function for output layer
    """
    A = 1/(1+np.exp(-Z))
    
    return A, Z

def relu(Z):
    """
    Returns Z if Z >= 0 else, 0
    """
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    
    return A, Z

def relu_bp(dA, Z):
    """
    Implement backprop for dA (dA/dZ = 1 if Z >= 0, 0 otherwise) at a single ReLu unit
    Return dZ 
    """
    dZ = np.array(dA, copy=True)  
    assert(dZ.shape == Z.shape)
    
    dZ[Z <= 0] = 0     # derivative of ReLu returns 0 if x < 0 and 1 if x >= 0 
    assert(dZ.shape == Z.shape)
    
    return dZ

def sigmoid_bp(dA, Z):
    """
    backprop for single sigmoid activation unit
    """
    
    A = 1/(1 + np.exp(-z))
    dZ = dA*A*(1-A)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
```



## **2. Random Initialization**
- this section, we will define 2 helper functions, first one is for intializing parameters for 2-layer model and second one extends this intializing process to L layers

### **2.1 Two-Layer Neural Network**

- The model's structure is: LINEAR (Wx + b) -> RELU (Activation function) -> LINEAR (Wx + b) -> SIGMOID (Activation function).
- Use np.random.randn(shape)*0.01 with the correct shape for random initialization of weight matrices (W).
- Use zero initialization for the biases (b). Use np.zeros(shape=())

```python
def init_params(nx, nh, ny):
    """
    Argument:
    nx : size of the input layer
    nh : size of the hidden layer
    ny : size of the output layer
    
    Returns:
    W1 : (nh, nx)
    b1 : (nh, 1)
    W2 : (ny, nh)
    b2 : (ny, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.rand(nh, nx)*0.01
    b1 = np.zeros(shape=(nh, 1))
    W2 = np.random.rand(ny, nh)*0.01
    b2 = np.zeros(shape=(ny, 1))
    
    assert(W1.shape == (nh, nx))
    assert(b1.shape == (nh, 1))
    assert(W2.shape == (ny, nh))
    assert(b2.shape == (ny, 1))
    
    params = {"W1" : W1,
              "b1" : b1,
              "W2" : W2,
              "b2" : b2}
    
    return params
```
```python
params = init_params(4, 5, 2)
for key, val in params.items():
    print("{0} : {1}".format(key, val))
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/92680829/172635253-191f5053-5da2-432d-962f-f1c3afe8faba.png">



### **2.2 L-layer Neural Network**
- initialization process for deep L-layer network is much more complex than shallow model as it has to keep track of the dimensions of all weights and bias matrices for all L-1 layers 
<img src="https://user-images.githubusercontent.com/92680829/172634409-6d0fb5f7-6b23-463f-969c-28a41e9ac499.png" width="900">

- so we will adapt for-loop to randomize parameters of each layer with the right dimension

```python
def init_params_L(dims):
    """
    Arguments
    dims : list taht contains the dimensions (n[i], n[i-1]) of every layer in network
    
    Returns
    params : python dict containing randomized initial parameters (W1, b1, W2, b2, ... , W[L-1], b[L-1])
    """
    
    np.random.seed(2)
    params = dict()
    L = len(dims)    # includes input layer (technically, L+1)
    
    for i in range(1, L):
        params["W{0}".format(i)] = np.random.rand(dims[i], dims[i-1])*0.01
        params["b{0}".format(i)] = np.zeros(shape=(dims[i], 1))
        
        assert(params["W{0}".format(i)].shape == (dims[i], dims[i-1]))
        assert(params["b{0}".format(i)].shape == (dims[i], 1))
        
    return params
```
```python
dims = [3, 4, 5, 2]    # nx : 3, nh1 : 4, nh2 : 5, nh3(output layer) : 2 
params = init_params_L(dims)

for key, val in params.items():
    print("{0} :\n {1}".format(key, val))
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/92680829/172852363-35d0ef02-4dbd-45cf-a9e4-6239f2e7c276.png">




## **3. Forward Propagation**
- Now, we've just initialized all of the parameters in L-model. 
- Next step, we will implement forward propagation modules that include 2 processes.
    - linear propagation : calculates Z[i] = W[i]*A[i-1] + b[i]
        - np.dot(W, A) + b 
    - linear-activation propagation : A[i] = Act_Func(Z[i])
        - RELU(Z) : Z if Z >= 0, else 0
        - Sigmoid(Z) : 1/(1 + np.exp(-Z))
- Finally, we will define a new helper functon that implements linear-activation propagation for every layer of our deep L-layer model at once


### **3.1 Linear Propagation**
```python
def linear_fp(A, W, b):
    """
    Arguments
    A : output of previous layer (n[i-1], m)
    W : weight matrix of current layer (n[i], n[i-1])
    b : bias matrix of current layer (n[i], 1)
    
    Returns
    Z : result of linear propagation = W*A + b
    cache : python dict containing A, W, b - stored for back-propagation
    
    """
    
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)
    
    return Z, cache
```
```python
A, W, b = linear_forward_test_case()   # see 1. Packages 
# A : (3, 2) 
# W : (1, 3)
# b : (1, 1)

Z, cache = linear_fp(A, W, b)
# expected Z shpae : (1, 2)

print("Z : {0}".format(Z))
```
<img width="270" alt="image" src="https://user-images.githubusercontent.com/92680829/172858019-47206ddb-6c3a-4918-9076-75ae547c7a4e.png">


### **3.2 Linear-Activation Propagation**
- this helper function calculates both linear and activation propagation
- here, we will use previously-defined activation function, sigmoid and Relu

```python
def linear_activation_fp(activation, A_prev, W, b):
    """
    Caculates both linear and activation propagation
    
    Arguments
    A_prev : output of previous layer (n[i-1], m)
    W : weight matrix of current layer (n[i], n[i-1])
    b : bias matrix of current layer (n[i], 1)
    
    Returns
    A : output of current layer (n[i], m)
    """
    
    Z, linear_cache = linear_fp(A_prev, W, b)   # linear_cache : A_prev, W, b
    activation_cache = Z
    
    if activation == "relu":
        A, _ = relu(Z)
    elif activation == "sigmoid":
        A, _ = sigmoid(Z)
        
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A, linear_cache, activation_cache
```
```python
A_prev, W, b = linear_activation_forward_test_case()

A, lin_cache, act_cache = linear_activation_fp("relu", A_prev, W, b)
print("--- ReLu Activation ---\nA : {0}\nZ (activation_cache) :\n {1}".format(A, act_cache))

print()

A, lin_cache, act_cache = linear_activation_fp("sigmoid", A_prev, W, b)
print("--- Sigmoid Activation ---\nA : {0}\nZ (activation_cache) :\n {1}".format(A, act_cache))
```
<img width="374" alt="image" src="https://user-images.githubusercontent.com/92680829/172862820-d8cda382-1c8d-4630-8c81-f9d6c457a3f9.png">


### **3.3 Forward Propagation for L-Layer model**
- Finally, we can implement previously defined linear_activatoin_fp function to every layer of our deep model at once using for-loop
- As an activaiton function, we will use relu for 1~L-1 layer and sigmoid for L layer, which is our final output layer
- Also, through this process, we will store all caches (A_prev, W, b and Z) from every layer into one list named as "caches" (results of fp for every L layer) 

```python
def L_model_fp(X, params):
    """
    Implement linear-activation forward propagation for L-layer model 
    Layer 1~L-1 : relu
    Layer L : sigmoid
    
    Arguments
    X : training examples (nx, m) 
    params : initialized params containing W1, b1 ~ W[L], b[L] 
    
    
    Returns 
    AL : final output from L layer
    caches : list of caches from every layer
             each cache has a form of (linear_cahce(A_prev, W, b), activation_cache(Z))
             index 0 ~ L-2 : activation as relu
             index L-1 : activation as sigmoid
    """
    
    caches = []
    L = len(params)//2 
    A_prev = X
    
    for i in range(1, L+1):
        if i == L:
            AL, lin_cache, act_cache = linear_activation_fp("sigmoid",
                                                            A_prev, 
                                                            params["W{0}".format(L)], 
                                                            params["b{0}".format(L)])
            caches.append((lin_cache, act_cache))
        else:
            A, lin_cache, act_cache = linear_activation_fp("relu", 
                                                           A_prev, 
                                                           params["W{0}".format(i)], 
                                                           params["b{0}".format(i)])
            caches.append((lin_cache, act_cache))
            A_prev = A
        
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches
```

```python
X, params = L_model_forward_test_case()  # X : (4, 2) / 2 layers
AL, caches = L_model_fp(X, params)

print("Final Ouptut AL : {0}".format(AL))

print()

for i, (lin, act) in enumerate(caches):
    print("-- Cache from Layer {0} --".format(i+1))
    print("A[{0}] :\n{2}\nW[{1}] :\n{3}\nb[{1}] :\n{4}".format(i, i+1, lin[0], lin[1], lin[2]))
    print("Z[{0}] :\n{1}".format(i+1, act))
    
print()

print("Length of Caches : {}".format(len(caches)))
```
<img width="500" alt="image" src="https://user-images.githubusercontent.com/92680829/173234205-6a151319-59b0-4be6-a73d-c4904d50bc33.png">



## **4. Cost Funciton** 
- Cost function is **Cross-Entropy Cost** that looks like below (same as we use all the time)<br/>
<img src="https://user-images.githubusercontent.com/92680829/173234463-3dad114c-2605-4bc2-907b-ed5cba8c68fb.png" width="400">

- let's make the helper function that computes cost with python

```python
def compute_cost(AL, Y):
    """
    Returns 
    cost : cross-entropy cost
    """
    
    m = Y.shape[1]
    
    cost = (-1/m)*np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))   # element-wise multiplication
    cost = np.squeeze(cost)    # make sure that cost has numeric value not matrix : eliminats axis whose size is 1
    assert(cost.shape == ())
    
    return cost
```
```python
Y, AL = compute_cost_test_case()

print("Cost for test case : {}".format(compute_cost(AL, Y)))
```
<img width="406" alt="image" src="https://user-images.githubusercontent.com/92680829/173234752-1caec6f9-20ac-40f6-98e0-50f64dd1b613.png">



## **5. Backward Propagation**
- Finally, we've built pretty much all helper functions including initializing parmaters, forward propagation and computing cost fucnton 
- One last left is Backwrad Propagation that is used to update paramters (W[l], b[l]) untill the model reaches to global optimum (at least close to it)
- here's the simplified diagram of backward propagation for L-layer model (2 layer in example)
<img src="https://user-images.githubusercontent.com/92680829/173235228-7891eccc-f641-4687-8e49-c985eec03bd1.png" width="700">

- There are largely three steps to propagate backwardly
    - LINEAR : dW[l], db[l], dA[l-1]
    - LINEAR -> ACTIVATION : dZ[l]
        - derivative of **Relu** funciton for 1~L-1 layer
        - derivative of **Sigmoid** function for L layer (output)

    <img src="https://user-images.githubusercontent.com/92680829/173235517-c3e9560f-5207-4913-a4df-232ac966e61c.png" width="280">
    <img src="https://user-images.githubusercontent.com/92680829/173235889-17bb7823-3ab2-4347-a1d1-3c03cf6fd7f5.png" width="280">

    - note that dZ[l] is needed to calculate dW[l], db[l], dA[l-1] -> **calculation of dZ[l] should precedes before dW[l], db[l], dA[l-1]**
   
### **5.1 Linear Backward Propagation**
- linear bp function computes derivative of Z[l] (W[l]*A[l-1] + b[l]) with respect to W[l], A[l-1], b[l]
- make sure that derivative should keep same dimension with its original matrix

```python
def linear_bp(dZ, cache):
    """
    Implement linear back-propagation for a single layer
    
    Arguments
    dZ (n_cur, m) : gradient of cost with respect to Z (lienar output)
                        gained from linear-activation backward
    cache : products from forward propagation containing (A_prev, W, b) and Z
    
    Returns
    dA_prev (n_prev, m), dW (n_cur, n_prev), db (n_cur, 1) : gradient of cost with respect to A_prev, W, b respectively
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m    # (n_cur, m) x (m, n_prev) = (n_cur, n_prev)
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True) / m)    # array that has length n_cur / axis = 0 along the row, 1 along the column
    dA_prev = np.dot(W.T, dZ)   # (n_prev, n_cur) x (n_cur, m) = (n_prev, m)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(len(db) == dZ.shape[0])
    
    return dA_prev, dW, db
```
```python
dZ, linear_cache = linear_backward_test_case()   

# dZ : (2, 2) / linear_cache - A_prev : (3, 2), W : (2, 3), b : (2, 1)

dA_prev, dW, db = linear_bp(dZ, linear_cache)

print("dA_prev :\n{0}".format(dA_prev))
print("dW :\n{0}".format(dW))
print("db :\n{0}".format(db))
```
<img width="400" alt="image" src="https://user-images.githubusercontent.com/92680829/173363672-7648b7d4-8381-4dd9-9b34-febbf4d7922f.png">


### **5.2 Linear-Activation Backward Propagation**
- We've built linear-backward propagation helper function for dW, dA_prev, db
- Now using this linear bp function and previously defined sigmoid and relu bp fucntions, we will write linear-activation backward propagation function, which computes two types of activation function 
    - Relu for 1~L-1 layer : dZ = 1 if Z > 0, else dZ = 0
        - dZ = relu_bp(dA, Z)
    - Sigmoid for L layer : dZ = A(1-A)
        - dZ = sigmoid_bp(dA, Z)
- order of back-propagation is **LINEAR-ACTIVATION** (dZ) -> **LINEAR** (dA_prev, dW, db)


```python
def linear_activation_bp(activation, dA, cache):
    """
    Implement relu-backward for 1~L-1 layer and sigmoid-backward for L layer (output) 
    
    Arguments
    dA : post-activation gradient of cost with respect to A (A for current layer)
    cache : tuple of caches (linear_cache, activation_cache) stored from linear-activation forward propagtion
    activation : type of activation function at current layer - define the form of dZ
    
    Returns
    dW : (n_cur, n_prev)
    dA_prev : (n_prev, m)
    db : list that has length of n_cur (squeezed to eliminate the axis of size 1)
    """
    
    linear_cache, Z = cache   # linear_cache, activation_cache
    
    if activation == "relu":
        dZ = relu_bp(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_bp(dA, Z)
        
    dA_prev, dW, db = linear_bp(dZ, linear_cache)

    return dA_prev, dW, db
```
```python
dA, cache = linear_activation_backward_test_case()   
# dA : (1, 2) / cache : (linear_cache(A, W, b), act_cache(Z))

dA_prev, dW, db = linear_activation_bp("relu", dA, cache)
print("-- Relu Activaiton --")
print("dA_prev :\n{0}".format(dA_prev))
print("dW :\n{0}".format(dW))
print("db :\n{0}".format(db))

print()

dA_prev, dW, db = linear_activation_bp("sigmoid", dA, cache)
print("-- Sigmoid Activaiton --")
print("dA_prev :\n{0}".format(dA_prev))
print("dW :\n{0}".format(dW))
print("db :\n{0}".format(db))
```
<img width="437" alt="image" src="https://user-images.githubusercontent.com/92680829/173371968-a97dcaf1-3f83-4ca2-b516-15b8b265fac2.png">



### **5.3 Backward Propagation for L-layer Model**
- Finally, we will implement the backward propagation for the whole network.
- we will use "caches" which is the list of caches from all layers that we've gained through the process of forward propagation
- Image below shows the simplified diagram of backward pass<br/>
<img src="https://user-images.githubusercontent.com/92680829/173373510-59b83205-7411-4232-abb6-9d48fd0d8699.png" width="450">

- before starting L-layer back-propagation, we need to calculate dA[L], which is the initial input of back-propagation
- dA[L] is the **drivative of Cost with respect to final forward-propagation output A[L]**
    - dA[L] = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))<br/>
    - you can easily prove this equation by taking partial derivative to our cross-entropy cost function with respect to AL
    <img src="https://user-images.githubusercontent.com/92680829/173375095-55679bc0-cb06-488d-ae98-757187b61b77.png" width="200">

```python
def L_model_bp(AL, Y, caches):
    """
    Implement backward propagation : 
    [LINEAR-ACTIVATION (sigmoid)] -> [LINEAR] -> ([LINEAR-ACTIVATION (relu)] -> [LINEAR]) * L-1
    
    Arguments 
    AL : initial input of bp (1, m), final post-activation output of forward propagation
    Y : true label (1, m), required here to derive dAL (-Y/AL + 1-Y/1-AL)
    caches : A_prev, W, b (linear_cache), Z (activation_cahce) from every layer, stored during forward propagation
    
    Returns
    grads : python dictionary with gradients of all parameters (dW[1], db[1] ... dW[L], db[1])
    """
    
    m = Y.shape[1]
    L = len(caches)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))   # (1, m)
    grads = dict()
    dA = dAL
    
    for i in range(1, L+1):
        if i == 1:
            dA_prev, dW, db = linear_activation_bp('sigmoid', dA, caches[L-i])
        else:
            dA_prev, dW, db = linear_activation_bp('relu', dA, caches[L-i])
    
        grads["dA{0}".format(L-i)] = dA_prev
        grads["dW{0}".format(L-(i-1))] = dW
        grads["db{0}".format(L-(i-1))] = db
        
        dA = dA_prev
        
    return grads
```
```python
AL, Y, caches = L_model_backward_test_case()   
# 2 Layer
# m : 2
# unit size of layer 1 : 3
# unit size of layer 2 : 1

grads = L_model_bp(AL, Y, caches)
L = len(caches)

for i in range(1, L+1):
    print("-- Layer {0} --".format(i))
    if i == 1:
        print("dX :\n{0}".format(grads["dA{0}".format(i-1)]))
    else : 
        print("dA{0} :\n{1}".format(i-1, grads["dA{0}".format(i-1)]))
    print("dW{0} :\n{1}".format(i, grads["dW{0}".format(i)]))
    print("db{0} :\n{1}".format(i, grads["db{0}".format(i)]))    
    print()
```
<img width="480" alt="image" src="https://user-images.githubusercontent.com/92680829/173830938-cf0e91f3-875d-4403-9598-46eff7cd41f1.png">




## **6. Update Parameters**
- Now it's almost done. Only one left is a function to update parameters with the gradient values from grads, which is a list of gradients of each parameter that we got from L_model_bp function 
- This step is called **"Gradient Descent"**, which means we repeatedly update paramters with its gradient against cost untill the model reaches to global optimum (gradient goes close to zero)
- We also need to set proper **α**, **learning rate** to adjust the speed of learning so that our algorithm doesn't diverge, but converge<br/>
 <img src="https://user-images.githubusercontent.com/92680829/173846929-3b40263b-62a9-4e02-9d38-8333636f520f.png" width="200"><br/>
<img src="https://user-images.githubusercontent.com/92680829/173847481-790d3a5f-8e84-44a6-8144-6caace55237c.png" width="550">


```python
def update_params(params, grads, lr):
    """
    Update parameters using gradient descent
    
    Arguments
    params : python dict containing your parameters
    grads : python dict containing gradients of all parameters
    lr : learning rate α
    
    Returns
    """
    
    L = len(params)//2
    
    for i in range(1, L+1):
        params["W{0}".format(i)] -= lr*grads["dW{0}".format(i)]
        params["b{0}".format(i)] -= lr*grads["db{0}".format(i)]
        
    return params 
```

---

- Congrats that we've finished all the functions required for building deep L-layer model (no matter how big it is!) step by step
- In the next practice, we will put all these fucntions together to build two types of models:
    - 2-layer neural network
    - L-layer neural network
- We will use these two models to classifiy cat vs non-cat images (as we did with logistic regression classifier) and compare the performance of two models