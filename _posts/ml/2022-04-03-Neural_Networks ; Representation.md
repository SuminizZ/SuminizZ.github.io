---
title : "[Coursera : ML Specialization] - Neural Networks : Representation"
categories : 
    - Machine Learning
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---


## **Why Deep Learning?**
- much better to learn complex hypotheses with large n (sample num)

## **Model Representation 1**
- **Biological Neuron**

<img src="https://user-images.githubusercontent.com/92680829/156725256-e58dd176-d9d9-46ae-8a81-f914e6d7a6ca.png" width="400">

- **Artificial neural network - computational representation of a neuron**
    - Example)
        - Logistic Unit
            - sigmoid activation function
                - activation function defines the type of the output of a neuron/node given an input or set of input
                - activation depends on
                    - The input(s) to the node
                    - The parameter associated with that node (from the Ɵ vector associated with that layer)

            - <img src="https://user-images.githubusercontent.com/92680829/156727186-545826c6-cb96-4ecf-bbb2-8f3c57bc99c8.png" width="400">
            
    - Neural Network
        - <img src="https://user-images.githubusercontent.com/92680829/156729290-b17642a6-adaa-4899-a730-a0390ffb25d8.png" width="400">
        - Terms
             - ai(j) - activation of unit i in layer j 
             - Ɵ(j) - matrix of parameters controlling the function mapping from layer j to layer j + 1
        - If network has 
            - sj units in layer j (the number of units in layer j, plus an additional unit) and sj+1 (the number of units in layer (j + 1)) units in layer j + 1 

- Example of neural networks and associated calculations
    <img src="https://user-images.githubusercontent.com/92680829/156949191-97a8f139-b671-4c1c-a865-2f42d83c1336.png" width="500">


    

- The activation value on each hidden unit (e.g. a12 ) is equal to the sigmoid function applied to the linear combination of inputs
    - a1(2) = g(sum of Ɵ10(1)x0 + Ɵ11(1)x1 + Ɵ12(1)x2 + Ɵ13(1)x3 
    - Ɵ(j) is a matrix of S(j+1) x (Sj + 1)
- Every input/activation goes to every node in following layer
    - Ɵjil
        - j (first of two subscript numbers)= ranges from 1 to the number of units in layer l+1
        - i (second of two subscript numbers) = ranges from 0 to the number of units in layer l
        - l is the layer you're moving FROM
        - <img src="https://user-images.githubusercontent.com/92680829/156950160-83e48940-e20a-4530-b562-f4214a8ad399.png" width="300">




---

## **Model representation II**


- Zi(j+1) = Ɵi0(j)x0 + Ɵi1(j)x1 + Ɵi2(j)x2 + Ɵi3(j)x3
    - S(j) = 0~3
- ai(j+1) = g(Zi(j+1))
- Z(j+1) = S(j+1) x 1 vector
- a1 = x
    - a1 is the activations in the input layer
    - Obviously the "activation" for the input layer is just the input!
- a1 is the vector of inputs
- a2 is the vector of values calculated by the g(z2) function

### **Forward Propagation**

<img src="https://user-images.githubusercontent.com/92680829/156951150-070b0cfc-f44e-46a5-83e6-47476418d745.png" width="700">
<img src="" width="1000">

- Layer 3 is a logistic regression node
    - The hypothesis output = g(Ɵ102 a02 + Ɵ112 a12 + Ɵ122 a22 + Ɵ132 a32)
    - g is just logistic regression 
    - **The only difference is, instead of input features, the features are just values calculated by the previous hidden layer**
    - The features a12, a22, and a32 are calculated/learned - not original features
    - So the mapping from layer 1 to layer 2 is determined by another set of parameters - Ɵ1
    - **So instead of being constrained by the original input features, a neural network can learn its own features to feed into logistic regression**
    - Flexibility to learn whatever features it wants to feed into the final logistic regression calculation
    - Here, we're letting the hidden layers do that, so we feed the hidden layers our input values, and let them learn whatever gives the best final result to feed into the final output layer


---

## **Non-Linear Classification Example (1) - XOR / XNOR**

- x1 XOR x2 : only TRUE when one of them (x1 or x2) is 1 (10 or 01)
- x1 XNOR x2 : only TRUE when 1. both of them are 1 or 2. both of them are 0 
- X mark 1 : XNOR / o mark 1 : XOR

- <img src="https://user-images.githubusercontent.com/92680829/156952865-aeec9b59-0f25-48e4-80f8-9dd7c1989ec4.png" width="500">

#### **x1 AND x2** 
- simple example of how neural networks can operate AND function with sigmoidal activation function
<img src="https://user-images.githubusercontent.com/92680829/156953394-798c316f-0ad9-4e30-8c83-b6e07d6dfa03.png" width="600" >

#### **x1 XNOR x2 Example**
- combination of x1 AND x2 (a1(2)) + x1 NAND x2 (a2(2)) 
- one input layer, one hidden layer, one output layer

- <img src="https://user-images.githubusercontent.com/92680829/156954991-dd0d5934-a909-4be1-828b-479c207d65cd.png" width="600" >

#### **x1 XOR x2**
- all same with the example above except below 
- Ɵ10(2) = 10 , Ɵ11(2) = -20 , Ɵ12(2) = -20
- if and only if : x1 XOR x2


---

## **Multi-Class Classification**



- Suppose that we have 4 classes to distinguish 
    - pedestrian, car, motorcycle, truck
    - how can neural network recognize these different classes? 
    - instead of assigning each classes numerical value like 1, 2, 3, 4 
    - we can assign each class 4 different 4 x 1 vector 
    - where each element of vector represents each class
<img src="https://user-images.githubusercontent.com/92680829/156956598-022e6279-58d0-43ea-996a-9d88ced4bddb.png" width="700">
<img src="" width="1000">
    - run all three classifiers on the input, and then pick the class i that maximizes the probability that hθ(i)(x) = 1