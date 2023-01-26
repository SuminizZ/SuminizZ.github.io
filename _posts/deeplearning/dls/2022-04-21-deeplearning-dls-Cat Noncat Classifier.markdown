---
layout: post
title : "[Neural Networks and Deep Learning] Practice : Cat/Non-Cat Classifier with Logistic Regression"
date: 2022-04-21 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-dls] 
tags: [Python, deep learning, Coursera, Neural Networks and Deep Learning]
# toc : true
# toc_sticky : true
---


<br/>

# **Cat/Non-Cat Classifier with Logistic Regression**

<br/>

## **Load Dataset**

```python
import numpy as np
import matplotlib.pyplot as plt
import h5py      # required for interacting with files stored on h5 file
import scipy     # for test
from PIL import Image
from scipy import ndimage

%matplotlib inline
```


```python
def load_dataset():
    with h5py.File('C:/Users\DNI_180902/Desktop/Data/DL-WK2/train_catvnoncat.h5', 'r') as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('C:/Users\DNI_180902/Desktop/Data/DL-WK2/test_catvnoncat.h5', 'r') as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```


```python
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```


```python
print("Training_Set_X : {0}".format(train_set_x_orig.shape))
print("Test_Set_X : {0}".format(test_set_x_orig.shape))

# (num of examples, (Height, Width px size of image), channels(RGB - 3 channels))
# Training_Set : 209 examples of (64, 64, 3) 3-Dimensional image data
# Test_Set : 50 examples of (64, 64, 3) 3-Dimensional image data
```

<img width="450" alt="image" src="https://user-images.githubusercontent.com/92680829/168480368-a0997047-c7d9-4a7b-b822-1cc395b0f2fd.png">


```python
print("Training_Set_Y : {0}".format(train_set_y.shape))
print("Test_Set_Y : {0}".format(test_set_y.shape))

# result : 1 (Cat) or 0 (Non-Cat)
# 209 training examples
# 50 test examples
```
<img width="284" alt="image" src="https://user-images.githubusercontent.com/92680829/168480534-d9f67abf-d558-450f-a528-6f719a2fe405.png">

<br/>

### **Reshape X Dataset**
- 209 training examples of 64x64x3 image (64x64x3, 209)
- <img src="https://user-images.githubusercontent.com/92680829/161663284-2172ce3e-d8ab-4d58-bc68-dc9c747f4615.png" width="400">


```python
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print(train_set_x_flatten.shape)
print(test_set_x_flatten.shape)

print(train_set_x_flatten[0:5,0])    # sample check
```

<img width="291" alt="image" src="https://user-images.githubusercontent.com/92680829/168480573-c357c0fc-f52b-4636-9561-c28c1b854cf0.png">


```python
# Normalization
# pixel range : 0~255 
# convert all the values of matrix ranging from 0 ~ 1

train_set_x_flatten =  train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255
print(train_set_x_flatten[0:5,0])
```
<img width="489" alt="image" src="https://user-images.githubusercontent.com/92680829/168480588-30c3b2ff-7d9b-4c9e-ab94-8d1fd41fc962.png">

<br/>

## **Building Learning Algorithm**

<br/>

### **General Architecture of the learning algorithm**

- <img src="https://user-images.githubusercontent.com/92680829/168478884-06b82252-bb1d-461d-902a-5a0347ea725b.png" width="500" >

- features : 12388 (64x6x3)
- activation function : logistic regression
- classifier : binary classifier (1 for cat, 0 for non-cat)
- decision-threshold : 0.5

- Mathematical expression of the algorithm:
    - <img src="https://user-images.githubusercontent.com/92680829/161677221-7ea2d0b9-27e6-4e1b-80d0-4b855ad58119.png"  width="400">
    - Cost is then computed by summing over all training examples:
        - <img src="https://user-images.githubusercontent.com/92680829/161677367-291b20f4-179f-43b7-a087-8578059f5251.png"  width="200">

<br/>

### **STEPS**
1. Define Model Structure
2. Initialize Parameters
3. Repeat Below
    - Calculate current loss : Forward Propagation
    - Caculate current gradient : Backward Propagation
    - Update parameters by gradient descent

<br/>

### **Helper functions**

<br/>

#### sigmoid function
```python

def sigmoid(z):    
    return 1/(1 + np.exp(-z))
```

<br/>

#### initialize parameters
```python
# dim : 64x64x3

def init_params(dim):     
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```

<br/>

#### Forward & Backward Propagation

```python
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, m)
    Y -- true "label" vector (0 non-cat, 1 cat) of size (1, m)
    
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """
    
    m = X.shape[1]

    # forward
    A = sigmoid(np.dot(w.T, X) + b) 
    cost = (-1/m)*(np.sum(Y*np.log(A)) + np.sum((1 - Y)*np.log(1 - A)))
    
    # backward
    dw = (1/m)*np.dot(X, (A - Y).T)
    db = (1/m)*np.sum(A - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    
    grads = {"dw" : dw,
            "db" : db}
    
    return grads, cost
```
<br/>

#### Optimization
```python
# Finding w that can minimizes Cost by Gradient Descent

def optimize(w, b, X, Y, num_iter, lr, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w, b, X, Y ~ same as above
    num_iter -- number of iterations of iteration loop
    lr -- regularization factor
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    costs = []
    cost = float('inf')
    nan_cnt = 0
    for k in range(num_iter):
        print_cost = False
        grads, cost_new = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w -= lr*dw
        b -= lr*db
        
        if not np.isnan(cost_new) and cost_new < cost:
            flag = 0
            best_w = w
            best_b = b
            cost = cost_new
            best_iter = k
        else:
            flag += 1
        
        if k%100 == 0:
            print_cost = True
        
        if print_cost:
            costs.append(cost_new)
            print("Cost After {0}th iterations : {1}".format(k, cost_new))
    
        if flag >= 2000:
            return (best_iter, cost, costs, best_w, best_b, w, b)

    return (best_iter, cost, costs, best_w, best_b, w, b)

```

<br/>

#### predict cat(1) or Non-Cat(1)
```python
def predict(best_w, best_b, X):
    '''
    Predict whether the label is 0 or 1 using optimized lr parameters (w, b)

    Returns:
    Y_prediction - a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    A = sigmoid(np.dot(best_w.T, X) + best_b)
    m = A.shape[1]
    pred = np.zeros((1, m))
    
    for i in range(m):
        if A[0, i] >= 0.5:
            pred[0, i] = 1
        else:
            pred[0, i] = 0
            
    assert(pred.shape==(1, m))
    
    return pred
```

<br/>

#### Merge all functions into Final Model
```python
def model(X_train, X_test, Y_train, Y_test, num_iter = 2000, lr = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (64*64*3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (64*64*3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    n_px = X_train.shape[0]
    w, b = init_params(n_px)
    
    print("< Train Dataset >")
    best_iter, cost, costs, best_w, best_b, w, b = optimize(w, b, X_train, Y_train, num_iter, lr, print_cost)
    
    w_test, b_test = init_params(n_px)
    print("< Test Dataset >")
    _, _, costs_test, _, _, _, _ = optimize(w_test, b_test, X_test, Y_test, num_iter, lr, print_cost)
    
    pred_train = predict(best_w, best_b, X_train)
    pred_test = predict(best_w, best_b, X_test)
    
    print("Train Accuracy : {0}".format((1-np.mean(np.abs(Y_train - pred_train)))*100))
    print("Test Accuracy : {0}".format((1-np.mean(np.abs(Y_test - pred_test)))*100))
    
    d = {"cost" : cost,
         "costs_test" : costs_test,
        "costs" : costs,
         "w" : w,
         "b" : b,
        "best_w" : best_w,
        "best_b" : best_b,
        "learning_rate" : lr,
        "num_iter": num_iter,
        "best_iter" : best_iter}
    
    return d
```

<br/>

## **Check Model Performace**

```python
d = model(train_set_x_flatten, test_set_x_flatten, train_set_y, test_set_y, 2000, 0.005, False)

# make sure that you set the appropriate learning rate
```

<img width="464" alt="image" src="https://user-images.githubusercontent.com/92680829/168480617-cfb5bf87-3ab9-4ca9-8f9b-d5ea3e5ef579.png">
<img width="457" alt="image" src="https://user-images.githubusercontent.com/92680829/168480667-d1da9b05-3671-4843-8d31-7df5120919ba.png">

```python
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(d['costs_test'], color='b', label="Test")
plt.plot(d['costs'], color='g', label="Train")
plt.legend(fontsize=15)
plt.xlabel("Iterations (per 100)", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.title("Learning Rate : 0.005", fontsize=15)
plt.show()
```

<img src="https://user-images.githubusercontent.com/92680829/168479802-dc0bc9fb-3a21-454d-841e-139b1e03def5.png" width="450">
    
<br/>

### **How Learning Curve Differs by Learning Rate**

```python
from collections import defaultdict

lrs = [0.01, 0.005, 0.001, 0.0005]
res_by_lr = defaultdict(list)

for lr in lrs:
    d = model(train_set_x_flatten, test_set_x_flatten, train_set_y, test_set_y, 2000, lr, False)
    res_by_lr["{0}".format(lr)] = d["costs"]
```

```python
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['r', 'g', 'b', 'violet']

for i, (lr, costs) in enumerate(res_by_lr.items()):
    plt.plot(costs, color=colors[i], label=lr)

plt.legend(fontsize=12)
plt.xlabel("Iterations (per 100)", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.title("Leaning Curves by Different Learning Rate", fontsize=15)
ax.set_facecolor('w')
plt.show()

# turns out 0.01 is the best learning rate 
```

<img src="https://user-images.githubusercontent.com/92680829/168479858-05752ae9-e075-4956-ba9e-5333c0dc4671.png" width="500">
    
<br/>

### **Test With Your Own Image : Cat or Not Cat?**
<img src="https://user-images.githubusercontent.com/92680829/161874013-9c9c3aeb-594a-46fb-b539-66e8458bcb34.png"  width="350">

```python
from PIL import Image
import matplotlib.image as img
from urllib.request import urlopen

def get_cat_img(url):
    my_cat = Image.open(urlopen(url))

    n_px = train_set_x_orig.shape[1]
    catarray = np.asarray(my_cat)
    cat = np.array(Image.fromarray(catarray).resize(size=(n_px, n_px))).reshape((1, -1)).T
    
    return cat
```


```python
# used model : lr - 0.01 

d = model(train_set_x_flatten, test_set_x_flatten, train_set_y, test_set_y, 2000, 0.01, False)
best_w, best_b = d["best_w"], d["best_b"]
```

```python
cats = ["https://user-images.githubusercontent.com/92680829/161882766-dfe0cd10-0ed3-4b87-9659-20c81c61b8b5.png",
       "https://user-images.githubusercontent.com/92680829/161882868-1753eeba-42fc-479c-94ab-f9c3d87529e7.png",
       "https://user-images.githubusercontent.com/92680829/161882894-904d0a81-1518-4b7b-82af-5329584b618b.png",
       "https://user-images.githubusercontent.com/92680829/161882954-f852ced5-5596-4286-bc23-df45797a4142.png",
       "https://user-images.githubusercontent.com/92680829/161883046-ab99cec6-2ce3-4d47-8e36-76085d929924.png",
       "https://user-images.githubusercontent.com/92680829/161892955-84c22938-2008-43f6-8f8f-8930508ec0ec.png",
       "https://user-images.githubusercontent.com/92680829/161892992-a5dca781-cfc0-48c3-89b6-2c5e90d322ef.png",
       "https://user-images.githubusercontent.com/92680829/161893118-f505be6f-6dee-43a2-8fae-743195d4e2b5.png",
       "https://user-images.githubusercontent.com/92680829/161893303-90588cc4-2e54-47e6-80ed-ed7360c4ec5f.png",
       "https://user-images.githubusercontent.com/92680829/161893353-6ec427d8-8789-46c0-b9ac-3627aafc95ab.png",
       "https://user-images.githubusercontent.com/92680829/161893642-c49cc4ee-7eb0-4321-8e93-2e7a253a9fca.png",
       "https://user-images.githubusercontent.com/92680829/161893546-f969f22a-1003-495e-a0d3-30bbb9023ebd.png"]

Y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]).reshape((1, -1))
m = Y.shape[1]
pred = np.zeros((m, 1)).T
for i in range(m):
    cat = get_cat_img(cats[i])
    pred[0, i] = predict(best_w, best_b, cat)
```


```python
accuracy = (1-np.mean(np.abs(Y-pred)))*100
accuracy

print("정답 : {0} / 예측값 : {1}\n정확도 : {2}".format(Y, pred, accuracy))
```
<img width="580" alt="image" src="https://user-images.githubusercontent.com/92680829/168480746-0d3f8cb6-4637-476f-b3fb-e5cc31c064b5.png">

- As logistic regression is not the best algorithm as an image classifier,
- the performance of model is not that great
- Later on, other algorithms with better formance on distinguishing images will be covered.
