---
layout : post
title : "[Loss Funcitons] MAE, MSE, MLE"
categories : 
    - [deeplearning-topics]
tag : [Entropy, deep learning, Loss Function, MAE, MSE, MLE]
toc : true
toc_sticky : true
---

<br/>

# Loss Funciton
- There are largely two paradigms to define Loss Function
    - Minimizing Error - MAE, MSE
    - Maximizing Likelihood - Maximum Likelihood Estimation (MLE)

## **- Error Minimization**
- minimizes the error between predicted value and actual value
- Mean Absolute Error (MAE), Mean Squared Error(MSE)
- Actually you'll be able to derive all these cost functions with a few modifications to MLE, which will be covered later at anoter posting

- To copmute error, first we need to define **"how to measure the magnitude of error"**
    - For this, We can use the concept of **"Norm", which is a measure of magnitude or length of vector**
    - We can also use this norm as a distance function 
    - There are two types of frequently used Norm<br/> :
    <img src="https://user-images.githubusercontent.com/92680829/175437870-a8a8132a-d7ef-4565-8e5a-7131aec71c26.png" width="420">

    - **Vector L1 Norm**
        - L1 norm is calculated as the **sum of the absolute vector values**, where the absolute value of a scalar uses the notation \|a1\|. 
        - In effect, the norm is a calculation of the **Manhattan distance** from the origin of the vector space.
        - For example, if you got data x=(1,2,3), y=(-1,2,4), then the distance of vector x, y with L1 norm is d(x,y) = \|1-(-1)\| + \|2-2\| + \|3-4\| = 2 + 0 + 1 = 3 
        
    - **Vector L2 norm**
        - L2 norm can be defined as the **square root of the inner product of a vector**.
        - It is also known as the Euclidean norm as it's used to calculate the **Euclidean distance** from the origin
        - d(x, y) with L2 norm is root(4 + 0 + 1) = root(5)
    
    <br/>

    <img src="https://user-images.githubusercontent.com/92680829/175439398-81058366-012a-4b4c-9d1f-817885bfbe7c.png" width="590">
    <br/>
    
    - Norm is also used in Regularization, which also will be explained at later posting

Now we've learned how we can measure the magnitude of error vector using L1 and L2 norm, let's apply this concept to understand what "MAE" and "MSE" are and how they differ from each other.


### 1. Mean Absolute Error (MAE)
- MAE is alternatively called as "L1 Loss" as it calculates **the mean of L1 norm of error vector**
- <img src="https://user-images.githubusercontent.com/92680829/175493501-ae492f95-1832-4ae9-b87e-80c5240edd27.png"  width="200">

- Absolute scalar difference (L1 loss) at certain data point<br/>
    <img src="https://user-images.githubusercontent.com/92680829/175494216-f83c6d56-5af0-4df7-a1d3-10c26493cb1d.png"  width="400">


### 2. Mean Squared Error(MSE)
- MSE, the L2 Loss fucntion, calculates the mean of squared L2 norm of error vector
- <img src="https://user-images.githubusercontent.com/92680829/175494526-cdcb8d63-d479-4b23-8bca-c255198003f9.png"  width="220">


- Basically, MSE calculates the area of square with the length of error (predicted - real)<br/>
    <img src="https://user-images.githubusercontent.com/92680829/175493918-9ba61ebd-7b76-4122-be48-62bfc7665fbd.png"  width="380">


### MAE vs MSE and RMSE (Root Mean Squared Error)
#### 1. Sensitivity to Outliers
- As each error component of MSE is squared, loss increases at a squared scale as the error increses, which makes MSE more sensitive to outliers.
- Thus, when a dataset has a large number of outlier cases, MAE is sometimes employed as an alternative to the MSE.
 
#### 2. Differentiability 
- MAE is not differentiable at its minimum where it forms cusp, thus you need to split the function into sections to calcualte the derivative. 
- MSE is differentiable at all data piont as it is quadratic function (its derivative exists in its entire domain)
    
    
#### **RMSE (Root Mean Squared Error)**
- square root of the MSE
    - <img src="https://user-images.githubusercontent.com/92680829/175776920-e7df2c48-ebf7-4728-b4a0-e6f74c78a45e.png"  width="320">
    
    
- RMSE is measured in the same units as the original data unlike MSE, thus **less sensitive to outliers** than MSE.
- RMSE doesn't use absolute value, so it is **differentiable at all data points** unlike MAE.
- But, since the errors are squared before they're averaged, the RMSE still gives a relatively high penalty to errors.
- Thus, RMSE is preffered over MAE when large errors are particularly undesirable
    

---

While both of the cost functions, MAE and MSE, are usually used for Regression problem, Cross-Entropy (CE) is widely used for Classification problem (binary or multi-classification)<br/>
This is because of the non-linearity of the model used for classification task, while regression uses linear model.<br/> 
Non-linearity causes MAE or MSE loss functions to become non-convex, which means gradient-based optimization cannot guarantee the convergence of model, which means the failure of learning. <br/>
(you can see the convexity of certain funciton by taking second derivative of that function, and see if its > 0) <br/>
To deal with this non-convex problems for non-linear model, we can alternatively use cross-entropy function<br/>
Next, we will understand what Maximum Likelihood Estimation (MLE) is, and derive Cross-Entropy Loss funciton by taking negative log to MLE.


## Maximum Likelihood Estimation (MLE)
- Unlike Error Minimization, MLE tries to **maximize a likelihood function** in order to find the probability distribution and parameters that best explain the observed data</br>


- Let me give you a simple example for intuitive understanding of MLE

- <img src="https://user-images.githubusercontent.com/92680829/175931377-fb6342bb-2191-466f-8a4f-cd4858406190.png"  width="500">
        
- which curve do you think can be the best fit for the observed data plotted as a histrogram ?
- it's pretty clear that the first curve (colored with blue, N(35, 5.5)) doesn't fit the data well, while the third (green) or fourth one (brown) seems to be the best fit, maximizing the likelihood that our data actually follows the distribution close to them. 
- Just like what we've done with the example above, finding the curve that can best expain the data is what MLE does


### Likelihood Function
- it describes the **joint probability of the observed data** as a function of the parameters of certain probability distribution 
- multiply the likelihood (height of the distribution) of each datapoint at the given statistical model 
    - likelihoods are the y-axis values for fixed data points with assumed distribution 
        <img src="https://user-images.githubusercontent.com/92680829/175931136-b024fa41-25fe-439f-9cd9-96573a185574.png" width="400" >
        
    - The likelihood, given multiple **independent events**, is the **product of the likelihoods of each of the independent events**
    - This follows from the definition of independence in probability: the probabilities of two independent events happening is the product of each probability.
  - <img src="https://user-images.githubusercontent.com/92680829/175221881-067d8464-1d27-4e33-8c6d-490942195d24.png"  width="180"> 
  

- now let's take a log to make **Log-likelihood fucntion**
    - <img src="https://user-images.githubusercontent.com/92680829/175222229-b654e7b8-3c41-4d43-9c30-426cdff105b3.png"  width="300"> 
  
    - As log is monotonically increasing, it only changes the scale, not alter the ranks of original data
    - **Advantages of taking Log**
        1. taking log can undo the exp, making it easier to compute
        2. can prevent arithmetic underflow which is a condition in a computer program where the result of calculation is more precise than the computer can actually compute. 
            - as the probaility is always smaller than 1, repeating multiplying some vlaue smaller than 1 can signify the underflow
            - but taking log can resolve this problem because the log of a product is the sum of the logs


### Maximum Likelihood Estimation (MLE)
- Finding the parameters of the probability distribution under which the observed data is most probable (or under which it has highest probability density)
- This practically means finding parameters with which likelihood function has maximum value
    - <img src="https://user-images.githubusercontent.com/92680829/175224140-53922a32-b11a-4413-87da-f684a829daca.png"  width="200">
  
  
- Process
    - First assume that the data comes from a certain distribution
    - Then randomly pick some parameters for that distribution
    - Then calculate the likelihood of the observed data under the assumed distribution
    - Then use **Optimization Algorithm** like Gradient Descent to find the best parameters and distribution that maximize the likelihood
        - by taking partial derivative to log-likelihood function with respect to parameters and find the point where it becomes zero
        - <img src="https://user-images.githubusercontent.com/92680829/175234783-998dbb41-43b8-4d3e-954a-27b9794adea6.png"  width="410">


--- 

Actually, maximizing log-likelihood function is identical to mimizing the negative log-likelihood function and solving negative log likelihood is closely related to solving Cross-Entropy function.

For next posting, we'll see how negative log likelihood acts as a flexible loss function that can cover all types of problems (Regression, Binary Classification, Multi-Classification).