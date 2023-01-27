---
layout : post
title : "[Coursera : ML Specialization] - Dealing with Large-Scale Machine Learning : SGD & Map Reduce"
date: 2022-04-13 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>

## **Learning with large datasets**
- One of best ways to get high performance is take a low bias algorithm and train it on a lot of data
- We saw that so long as you feed an algorithm lots of data, they all perform pretty similarly
- However, simply increasing the size of data cannot always gaurantee good performance of model
    - <img src="https://user-images.githubusercontent.com/92680829/158741191-89aa8267-e785-44a5-96cc-0cbcfe3c36f6.png" width="300" >
<br/>

- Suppose you have a huge dataset with m greater than 1 bilion, 
- then if you are using linear regression, it means that you have to calculate (update) more than 1 bilion times per each learning step
- this will cause sever compuational cost
- then, how can you deal with this situation
    - 1. you have to check if we can train on 1000 examples instead of 100 000 000
         - Randomly pick a small selection
    - Then, plot the **Learniing Curves of Jcv and J train** according to the training size and figure out the issue that you're dealing with
        - **High Variance** : seems like add more training examples will help 
            - <img src="https://user-images.githubusercontent.com/92680829/158741783-290abb11-5614-46eb-8fcb-8c3965a6e227.png"  width="200">
            
        - **High Bias** : increasing the size of training examples will not gonna help to improve your model
            - <img src="https://user-images.githubusercontent.com/92680829/158741846-ef5cfc23-89ee-4e8d-8613-b406890580c1.png"  width="200">

<br/>

- Therefore, the most reliable ways to get a high performance machine learning system is to take a **low bias algorithm and train on a massive data set**
- How to deal with large scale dataset
    - 1. Stochastic Gradient Descent : efficiency
    - 2. Map Reduce

---
<br/>

## **Stochastic Gradient Descent**
<br/>

- When you have very large dataset, gradient descent becomes computationally very expensive
- to solve this issue, modifications to gradient descent is applied, which is called "Stochastic Gradient Descent"
- Previous version of Training Linear Regression with **Gradient Descent**
    - <img src="https://user-images.githubusercontent.com/92680829/158743275-2088e588-77f4-469d-948c-c5eec4480188.png"  width="500">
<br/>

- we have to repeat this updating process for **entire m** per one iteration
- this kind of gradient descent is called **Batch gradient descent**
    - not a effective choice for large dataset to train : LONG time to converge
- Instead of this, we will use different algorithm that doesn't require to see all of the training examples for every update 

    - <img src="https://user-images.githubusercontent.com/92680829/158744213-5e4c6eca-a3aa-427c-9cf3-18ff1e451b22.png"  width="320">
<br/>

- So the function represents the SGD calculates **cost of θj with respect to a specific single example (xi, yi)**
- measure how well is my hypothesis doing on a single example

<br/>

### **Algorithm**

1. Randomly shuffle dataset
    - means we ensure the data is in a random order so we don't bias the movement
        - speed up convergence a little bit
2. Loop (about 1~10 times)
    - <img src="https://user-images.githubusercontent.com/92680829/158745937-b9bdcfcc-6b2e-4816-b75d-a1b90d6aa988.png"  width="250">
    

- Can you see the difference here?
- gradient descent requires to compute the derivatives of all training example to update θ once!
    - On the other hand, SGD only takes a single training example x(i) for updating θj onece
    - and repeat updating for 1 ~ m examples 
    - Means we update the parameters on EVERY step through single data, instead of updating at the end of each loop through all the data

- Pattern of Convergence is different 
    - <img src="https://user-images.githubusercontent.com/92680829/158748465-5e302586-7b60-4960-b7a4-bd43a121bbce.png"  width="330">

- red line shows the updating track of parameters from batch gradient descent (**1 Update per Batch**)
- pink line is from SGD (**1 Update per 1 data**)
    - seems every update is slightly titled to every single data point, but converge to the optima at a much faster rate
        - Not necessarily decrease Jtrain for every update (even with well-tuned learning rate)
    - variant but faster
    - May need to loop over the entire dataset 1-10 times
        - If you have a truly massive dataset it's possible that by the time you've taken a single pass through the dataset you may already have a perfectly good hypothesis
    

- Due to its variance, SGD never actually converges like batch gradient descent does, but ends up wandering around some region close to the global minimum
    - In practice, this isn't a problem - as long as you're close enough to the global minimum

<br/>

### **Stochastic gradient descent convergence**
<br/>

- how can you be certain that your sgd has convergd to global minimum (at least close)
- how do you tune your learning rate α? 

- **Checking Convergence : Plot cost(θ, (xi, yi)), averaged over N examples**
    - 1. decreasing learning rate (upper left)
        - slower the convergence
        - but obtain slightly better cost (negligible sometimes)
    - 2. increasing N (>= 5000) (upper right)
        - also takes more time to plot (longer time to get single plotting point)
        - can smoothen the cost line
    - 3. small N (lower right)
        - line will fluctuate too much, preventing you from seeing actual trend 
        - if you elevate N, then you can see what's actually going on 
    - 4. increasing cost : diverging (lower right)
        - it shows that your algorithm fails to converge to minimum, (fails to find optimal parameters)
        - you should adjust your learning rate smaller, so that it can converge
    
    - <img src="https://user-images.githubusercontent.com/92680829/158756682-e98ca189-c71e-48a8-929a-9718bbb3967b.png"  width="500" >
        
<br/>

- **Learning rate (α)**
    - ! typically, α helds constant through entire learning process
    - but, you can also slowly decrease α over time (if you want the model to converge better)
        - **α = const1/(iterationNumber + const2)**
        - but you have take additional time to decide what const1 and const2 are
        - this means you're guaranteed to converge somewhere rathter than oscillating around it

- SGD can be a good algorithm for online learning where there's a great influx of data per second (massive training examples)
    - SGD will boost up your journey to find best parameters that will help your business decision

---
<br/>

## **Mini-Batch Gradient Descent**
- Compromise between Batch gradient descent & Stochastic gradient descent
    - Batch gradient descent: Use all m examples in each iteration (update)
    - Stochastic gradient descent: Use 1 example in each iteration
    - **Mini-batch gradient descent**: Use b examples in each iteration (b = mini-batch size , b <= m)
- Can work better than SGD in some cases

- **Algorithm**
    - needs to update 100 times with mini-batch size 10 and total data size 1000
    - <img src="https://user-images.githubusercontent.com/92680829/158750901-d4aad284-9fee-43c6-aefd-5881c43ce8da.png"  width="300">

<br/>

### **Mini-batch vs. stochastic**

- Advantage
    - Allows **Vectorized implementation**
        - each sum of b examples can be performed in a vectorized way 
        - can temporarily parallelize your computation (i.e. do 10 at once)
        - much more efficient to compute rather than computing all examples just as a single number
    
- Disadvantage
    - Optimization process to decide parameter b
        - But this is often worth it!

- Actually, Stochastic gradient descent (b=1) and Batch gradient descent (b=m) are just specific forms of batch-gradient descent

---
<br/>

## **Map Reduce and Data Parallelism**
<br/>

- Sometimes you have so massive data that you can't even handle all of them in one computer, no matter what algorithms you choose to use (even SGD)
- Some says that Map Reduce is equally or even more important than SGD !!
<br/>

####  **Example**
- Assume we are training Mini-Batch Algorithms with massive data (commonly greater than 4 bilions)
    - <img src="https://user-images.githubusercontent.com/92680829/158761990-651aeeda-8892-46a6-a68a-a1711a73b69a.png" width="600" >

- training all these examples at one computer will take too much time and cost
- So, we split these examples into (say) 4 Computers : Parallelising over different computers
    - Machine 1 : temp1 : Σ use (x1, y1), ..., (x100, y100) 
    - Machine 2 : temp2 : Σ use (x101, y101), ..., (x200, y200) 
    - Machine 3 : temp3 : Σuse (x201, y201), ..., (x300, y300) 
    - Machine 4 : temp4 : Σ use (x301, y301), ..., (x400, y400) 
- And then, we will gonna send all four result (temp 1~4) from each computer to one centralized computer where the actual update occurs
    - Put them together, and Update θ using 
        - <img src="https://user-images.githubusercontent.com/92680829/158762720-5dde4100-af27-4b68-92c8-9183925a4330.png" width="500" >

- Same optimization algorithm can be applied to Logistic Regression (if remember, partial derivative term of cost function of logistic and linear regression is same : sum over training set)

<br/>

### **Scheme of how map reduce happens**
- <img src="https://user-images.githubusercontent.com/92680829/158763099-38384558-f303-4ee1-be86-9d645425583a.png" width="500" >

- The bulk of the work in gradient descent is actually the summation of functions over training set
- the reason why batch gd takes more time to sgd is because it calculates the sum of larger data
- With Map Reduce Approach, each of the computers does a quarter of the work at the same time, so you get a 4x speedup
    - Of course, in practice, because of network latency, combining results, it's slightly less than 4x, but still good!
    
- Parallelization can come from
    - Multiple machines, CPUs, cores in each CPU
- So even on a single compute can you implement parallelization, which is called "Local Parallelization"
- with Hadoop : Open source implementation of Map reduce
