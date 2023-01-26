---
title : "[Coursera : ML Specialization] - Support Vector Machine (SVM)"
categories : 
    - Machine Learning
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

## **SVM cost functions from logistic regression cost functions**
- note that logistic regression has distinct cost function like below
    - <img src="https://user-images.githubusercontent.com/92680829/157171544-9b3bea85-5c75-4b5c-adec-52c3c8785bfd.png" width="450">
    <img src="" width="100%">

- To build a SVM we must redefine our cost functions
    - When y = 1 : **Cost1(z)**
        - Take the y = 1 function and create a new cost function
        - **Create two straight lines (magenta) which acts as an approximation** to the logistic regression y = 1 function

            - <img src="https://user-images.githubusercontent.com/92680829/157171844-73a76594-5922-48d6-a424-a38c50727076.png" width="300" >
        
    - When y = 0 : **Cost0(z)**
        - Do the equivalent with the y=0 function plot
        - <img src="https://user-images.githubusercontent.com/92680829/157171979-aaf355af-f7f7-457f-9a32-e4b92e6635fa.png" width="300" >


- For the SVM we take our two logistic regression y=1 and y=0 terms described previously and replace with
    - y = 1 : cost1(θT x)
    - y = 0 : cost0(θT x)
- So we get
    - <img src="https://user-images.githubusercontent.com/92680829/157172972-7e934335-3822-41b5-bd96-5b83ec6d52da.png" width="600">

### **Notational Difference between SVM and LR**
1. Get rid of the 1/m terms
    - doesn't affect optimal value where cost become minimum
2.  A + λB  ->  CA + B
    - Training data set term (i.e. that we sum over m) = A
    - Regularization term (i.e. that we sum over n) = B
    - So we could describe it as A + λB in logistic regression
    - For SVMs the convention is to use a different parameter called **C**
        - **C equal to 1/λ** and the two functions (CA + B and A + λB) would give the same value
        
- So, the final form is 
    - <img src="https://user-images.githubusercontent.com/92680829/157174060-ca10c16e-fa58-4b4a-a051-19c7427c6c30.png" width="600">


---

# **Large Margin Classification**

- SVM = Large Margin Classifier

- <img src="https://user-images.githubusercontent.com/92680829/157175054-019750e6-5311-42c8-8333-c8075e4203f8.png" width="600">

- SVM Decision Boundary become more strict
    - To be y = 1 : z >= 0 (LR) --> z >= 1 (SVM)
    - To be y = 0 : z < 0 (LR) --> z <= -1 (SVM)

---

## **SVM Decision Boundary**

<img src="https://user-images.githubusercontent.com/92680829/157176174-f89fa03b-e1a3-4b63-a682-c4bb4ccffa4e.png" width="500">
<img src="" width="100%" >

- **C** : controls trade-off between smooth decision boundary (greater margin, training error ↑) and hard decision boundary (small margin, test error ↑)
    - can control the size of margin (the distance between the dcb and each data point)
    - Large C : 
        - suppose that our C is huge, to minimize the cost function, margin needs to closer to zero, so it barely allows any error, which is called "Hard Decision Boundary" 
        - to draw the dcb that has small error from all data point, the decision boundary become more complex and thus, overfitted
        - small training error, but too large C can cause overfitting, thus large test error
    - Small C :
        - becasue of the small C, relatively big margin (error) is allowed, which is called "Smooth Decision Boundary"
        - more outliers are allowed 
        - can possibly cause under-fitting, and increase training error 

    - <img src="https://user-images.githubusercontent.com/92680829/157185717-de72db56-34e3-4984-a247-8f4169ffd978.png" width="450">

- SVM draws the decision boundary line that has the greatest margin 
    - <img src="https://user-images.githubusercontent.com/92680829/157176944-ab926fa4-0e3c-4ad0-af27-44925e4bf4cf.png" width="300">

### **Large Margin Classifier in the presence of outlier**
- how sensitive does the SVM respond to outliers 
    - large C : sensitive --> tend to include outliers to minimize the errors
    - small C : insensitive --> tend to ignore outliers


---

## **Mathmatics Behind Large Margin Classification**

### **SVM Decision Boundary**
- if C is really small, we can ignore the error part of cost function, so we only have to focus on the weights to minimize the cost

- <img src="https://user-images.githubusercontent.com/92680829/157561845-e825fad2-d4cc-4586-98d5-63ac7f2b704a.png" width="350">

- Two Simplifications 
    1. θ0 = 0
    2. θ1, θ2 only 2 features exist

- min(1/2 * sigma(θ^2)) = 1/2(θ1^2 + θ2^2) = 1/2*(root(θ1^2 + θ2^2))^2 = 1/2*[θ]^2
    - So, finally, this means our optimization function can be re-defined as 
        - <img src="https://user-images.githubusercontent.com/92680829/157562505-17d7635c-c4ea-413f-9c26-40b9bce36e1f.png" width="150" >
        - So the optimization objective of SVM is minimizing the squared norm


   <img src="https://user-images.githubusercontent.com/92680829/157563382-e7090e72-4ded-4b55-8d0d-614308c9ce69.png" width="700" >

- x(i) is one set of training example that consists of x1(1) and x2(1) 
    - Given our previous discussion about inner vector (uT.v = p[u])
    - **θT.x(i) = z = p*[θ]**
    - note ** p is the length of the projection from x(i) vector onto θ vector  


#### **SVM optimization algorithm aims to maximize margin to minimize the cost**
- <img src="https://user-images.githubusercontent.com/92680829/157597209-75991950-c072-4629-ace0-4edc12ebbba2.png" width="350" >
- if margin is small, then the value of p(i) also gets smaller, which makes [θ] larger to meet the condition for being classified
    - pi * [θ] >= 1 if y = 1
    - pi * [θ] <= -1 if y = 0
- otherwise, if margin is large, p(i) gets large and [θ] become smaller
    - <img src="https://user-images.githubusercontent.com/92680829/157597882-81ae2e79-e9c2-4e2e-ad2a-5b511059d690.png" width="650" >

- if [θ] is large, cost function (1/2[θ]^2) can't be minimized.
- Therefore, **by maximizing the value of p (maximizing the margin between dcb and the nearest data point), we can minimize the [θ], and eventually, Cost Function**
- SVM finds the best hyperplane(which in 2D, simply a line) that best separates the data points into proper classification, by maximizing the margin from dcb to tags.

---

### **-- Vector Inner Product --**
- <img src="https://user-images.githubusercontent.com/92680829/157201236-e91c6126-cd02-4d3d-96b0-7b4892e10f7f.png" width="650" >

- px[u]=uT x v
    - p=[v]cosx (cosx = uv/[u][v])
    - p=uv/[u] -> px[u] = uv (inner component of vector u and v)
    - u.v (u vector v vector) = uT x v
    - if the angle between v and u vector is greater than 90, p will have negative value

### **-- Why is θ is Perpendicular to the Decision Boundary? --**

- in SVM, decision boundary looks like 1. θT*(x)+b >= 1 or 2. θT*(x)+b <= -1  (b = x0, can be included as an element of vector) 
- if you set b-1 = c and b+1 = d, Then, you can change it to 
    - θT*(x)+c = 0 and 2. θT*(x)+d = 0
    - **doesn't really matter what the constant value is.**
- Pick a point x1 on the decision boundary (let's say constant value as k). We know:
    - θT(x−x1)+k=0
- Pick a point x2. We have:
    - θT(x−x2)+k=0
- Subtracting eqn1 from eqn2:
    - θT(x1−x2) =0 (when inner product of two vector is 0, the angle between them is 90, cos(90) = 0)
    - Then, θT vector is perpendicular to (x1-x2) vector !
    - and (x1-x2) vector is on the decision boundary (as both of them lies on dcb)
    - Finally, θT is perpendicular to decision boundary!!! 



---


# **Kernels**

## **1. Adapting SVM to Complex Non-Linear Cclassifiers**

- Issue 

    <img src="https://user-images.githubusercontent.com/92680829/157603239-c67876e2-f978-4b4c-b3b8-9090f679d3c4.png" width="600" >
    

    

- Instead of these high polynomial features, is there any better choice of features?

### **Given x, compute new feature depending on proximity to landmarks**
- These points l1, l2, and l3, were chosen manually and are called landmarks
- <img src="https://user-images.githubusercontent.com/92680829/157604990-0b91c8ab-db36-40ec-8525-03f7c3406b0e.png" width="600" >
- the function to get the similarity between x and l(i) is called **"Kernel"**
- and here, the type of the kernel is **"Gaussian kernels"** : 
    - exp(- ([x - l1]2 ) / 2σ2)






### **So, What these Kernels actually Do?**
<img src="https://user-images.githubusercontent.com/92680829/157607210-6c121f59-be40-4b23-80a4-a819cbc22dba.png" width="650" >

- defines the similarity between each x data point and selected landmark with the value within 0 ~ 1
- k(x, l) =: 0  -->  very far
- k(x, 1) =: 1  -->  very close
- you can get new features f(i) that represents the similarity between every x and landmark (i)

- If we plot f1 vs the kernel function of x1 and x2, we get a plot like this
    - l(1) = [3, 5]
    <img src="https://user-images.githubusercontent.com/92680829/157608069-e66eaa2d-5215-4993-be6b-e4cff8c4bfbb.png" width="300" >
- σ= 1 here
- Notice that when x = [3,5] then f1 = 1
- As each x moves away from [3,5] then the feature takes on values close to zero
- So this measures how close x is to this landmark !!

### **What does σ do?**
- σ2 is a parameter of the Gaussian kernel
- it defines the steepness of the rise around the contour
    - greater σ, less steeper contour
- Above example σ2 = 0.5

    - <img src="https://user-images.githubusercontent.com/92680829/157608742-ebc10ad6-9d17-4714-9fd1-20a5cda50f2e.png" width="300">


- can see the steepness of the rise is quite sharp
- You can see here that as you move away from [3,5] (same as l(1)) the feature f1 falls to zero much rapidly


- The inverse can be seen if σ2 = 3
    - <img src="https://user-images.githubusercontent.com/92680829/157608947-d5cdddc3-d1d6-4fbd-bbf8-b6e6e7ddd5d8.png" width="300" >

### **Application of Kernel to Classify Data**

<img src="https://user-images.githubusercontent.com/92680829/157611047-31ba9a1f-e9c4-4a14-8c78-563d55322b17.png" width="800" >

- Let's say that θ0 = -0.5, θ1 = 1, θ2 = 1, θ3 = 0 
    - it's multivariate logistic regression, where y = 1 when z >= 0 and y=0 when z < 0
    - tags that are close to l(1) and l(2) are classified to y = 1
        - f1 or f2 =: 1 / f3 = 0
        - positive z value (-0.5 + 0 + 1 + 0 = 0.5) --> classified to '1' 
    - but, tags that are close to l(3) are classified to y = 0
        - f1 and f2 =: 0 / f3 = 1
        - negative z value (-0.5 + 0 + 0 + 0) --> classified to '0' 

- Therefore, we now have decision boundary based on the proximity with the selected landmarks

- This is the example of how kernels helps classification by providing new features called f(i)

---

## **2. SVM with Kernels In Practice**

### **How to choose landmarks?**
<img src="https://user-images.githubusercontent.com/92680829/157614388-0850f51d-e7d9-4b8c-9a10-3dbe9fa005bc.png" width="600">

- 1) if you set landmark for every training examples (x(1) ~ x(m))
    - the location of every training examples itself and their proximity with others became the features
    - you can have all new features, measure of proximity between xi and all other xs
        - f1i, = k(xi, l1)
        - f2i, = k(xi, l2)
        - ...
        - fmi, = k(xi, lm)
    - In training, one of them necessarily have value 1 (when l(i) == x(i), fii = 1)
    - now we have **new feature vector [f1, f2, ... , fm] instead of [x1, x2, ..., xm]** (you can add x0 for bias, [m+1. 1] vector in that case) 
    - when you get the test data, model will calculate the proximity between that input and all other training examples (f1 ~ fm)
    

### **SVM hypothesis prediction with kernels**
- Predict y = 1 if (θT f) >= 0
    - Because θ = [m+1 x 1] 
    - And f = [m +1 x 1] 
- So, this is how you make a prediction assuming you already have θ

### **How can you get θ ?**
- same as the previous SVM learning algorithm from logistic regression
    -  <img src="https://user-images.githubusercontent.com/92680829/157617175-7f652b4a-7f25-43c0-94ee-076b92aaf831.png" width="700">

- Now, we minimize using f as the feature vector instead of x
- By solving this minimization problem you get the parameters for your SVM
- in case of this, n equals to m (as n is determined by the size of f (m) <-- x)

- when C is small, the cost function almost acts like below
    - <img src="https://user-images.githubusercontent.com/92680829/157619365-60366ecd-3967-414d-b429-06b9992a9006.png" width="150" >

- What many implementations do is (final mathmatical detail)

    <img src="https://user-images.githubusercontent.com/92680829/157619907-08bcd9de-ed12-456a-9167-071abd91a57e.png" width="100">
    <img src="" width="100%">
    
    - the matrix M depends on the kernel you use
    - Gives a slightly different minimization - means we determine a rescaled version of θ
    - Allows more efficient computation, and scale to much bigger training sets
    - If you have a training set with 10 000 values, means you get 10 000 features and Solving for all these parameters can become expensive
    - So by adding this in we avoid a for loop and use a matrix multiplication algorithm instead 

### **The Role of Parameters in SVM with Kernels**
<img src="https://user-images.githubusercontent.com/92680829/157620536-be9859eb-212f-4ac8-bdfd-9ff6fe5a4e25.png"  width="500">


