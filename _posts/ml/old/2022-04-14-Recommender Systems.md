---
layout : post
title : "[Coursera : ML Specialization] - Recommender System"
date: 2022-04-14 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [dml-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---

<br/>

## **Problem Formulation**
- <img src="https://user-images.githubusercontent.com/92680829/158548925-9de07576-9a00-4759-8cf3-bb82f43efd99.png" width="500" >

- Notations
    - nu - Number of users (called ?nu occasionally as we can't subscript in superscript)
    - nm - Number of movies
    - r(i, j) - 1 if user j has rated movie i (i.e. bitmap)
    - y(i,j) - rating given by user j to move i (defined only if r(i,j) = 1
    - mj - Number of movies rated by the user (j)

---

<br/>

## **Content-Based Recommenation**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/158550480-9412e2ac-62eb-434a-852c-4544b5a35e78.png" width="620" >

- x(i) denotes ith the 3x1 feature vector of ith movie
    - [intercept x0 = 1, x1(romance), x2(action)]
- we have {x(1), x(2), x(3), x(4), x(5)} for all 5 movies
- now, for this dataset, 
    - n = 2 (number of features)
    - m = 5 (number of examples)


- For each user j, we have to learn parameters Θ(j) [3x1] vector
    - by using Θ(j), we predict user (j)'s ratings of movie (i)
        - (θj)T xi = stars (rating)
    - For example, lets take user 1 (Alice) and see what ratings she will give for the movie, Cute Puppies of Love (x(3))
        - x(3)T = [1, 0.99, 0]
        - We have some parameter vector (θ1T = [0, 5, 0] ) associated with Alice (how these values are derived will be explained later)
        - now, we can calculate the predicted ratings as **θ1T.x(3) = 4.95**
        

<br/>

### **How to learn (θj)**

<br/>

- To learn θj for j in all user (1~j)
    - <img src="https://user-images.githubusercontent.com/92680829/158713848-5f22125e-d342-4bca-8c2b-e808de0875a7.png" width="650" >

- i:r(i, j) = 1 .. only if r(i, j) = 1 (only if the user actually rated the movie)
- we can delete the divider mj for simplification, as it has no effect on the result
- we don't panalize θ0, as you can see the regularization term starts from k = 1 (not 0)
- it equlas to the cost function of multivariate linear regression to find the θ that can minimizes the cost (error of prediction)

- **Gradient Descent Update**
    - Repeat 
        - <img src="https://user-images.githubusercontent.com/92680829/158714263-ae15a6f0-468a-44ba-a9a5-0fe12388bd34.png" width="610" >

- We just predicted the ratings of movies from user j based on the information of movie as x(i)
- This is why we call this type of learning **"Content Based Learning"**
- x(i) includes the information that how much this specific movie is related to each genres of movie such as romance, action, and etc.
- However, in practice. these information about movies are not really available all the time. 
- Therefore, next we'll gonna discuss about **"Non-Contents Based Approach"**

<br/>

### **User-Based Learning**

<br/>

- previously, we calculated θj based on x(i)
- but now, we will caculate x(i) based on the given θj, which is the preferences of user j for all features(genre)
- We must minimize an optimization function which tries to identify the best parameter vector associated with a **film**, not user
- So, the cost function can be 
    - <img src="https://user-images.githubusercontent.com/92680829/158716844-d54039fb-2380-435b-a2dd-b506fbd0fa9a.png" width="580" >

---

<br/>

## **Collaborative Filtering**

- Here we combine the ideas from before to build a **Collaborative Filtering Algorithm**
- Our starting point is as follows

    1. If we're given the **film's features** we can use that to work out the **users' preference**
        - <img src="https://user-images.githubusercontent.com/92680829/158717060-923f0d2a-5613-4d28-acbc-3754b479ae46.png" width="550" >
    
    2. If we're given the **users' preferences** we can use them to work out the **film's features**
        - <img src="https://user-images.githubusercontent.com/92680829/158716844-d54039fb-2380-435b-a2dd-b506fbd0fa9a.png" width="550" >

- As each user gives ratings to multiple movies and each movie is rated by multiple users 
- So we go back and forth to collaboratively gain the information about both users and movies

- Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.
- It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user

<br/>

### **Algorithm**

<br/>

- So basically, what we're gonna do is just to mix both of the algorithms (1. and 2.)
    - <img src="https://user-images.githubusercontent.com/92680829/158718603-9b8086ff-07f7-4ca9-839f-398592964048.png" width="700" >

- The squared error term is the same as the squared error term in the two individual objectives above
- So it's summing over every movie rated by every users
    - x(i) for i in (1 - nm) and θj for j in (1 - nu)
    - Sum over all pairs (i,j) for which r(i,j) is equal to 1
    

- This newly defined function has the property that
    - If you held x constant and only solved θ then you solve the, "Given x, solve θ" objective above
    - Similarly, if you held θ constant you could solve x
    - Only difference between this in the **back-and-forward** approach is that we **minimize with respect to both x and θ simultaneously**

**Algorithm Structure**

1. Initialize (θ1, ..., θnu) and (x1, ..., xnm) to small random values
    - A bit like neural networks - initialize all parameters to small random numbers
    - for symmetry breaking
2. Minimize cost function (J(x1, ..., xnm, θ1, ...,θnu) using gradient descent (or other advanced optimization algorithm)
    - We find that the update rules look like this
    - Upadting parameters for every movie as well as every users
    - <img src="https://user-images.githubusercontent.com/92680829/158729102-fb83f8d5-c126-45e6-936e-a3040c02c23f.png" width="600" >

- With all these parameters updated to minimize the cost, we can train the model to predict ratings of the movie that has not been rated by a user

---

<br/>

## **Vectorization : Low Rank Matrix Factorization**

<br/>

- predicted ratings 
    - <img src="https://user-images.githubusercontent.com/92680829/158731658-b8627a28-42cf-4749-834e-b8e95b028d41.png" width="450" >

- θjTx(i) : ratings of x(i) movie from user j
- Define Feature [nm x 1] matrix X : information about all movies 1 ~ nm
    - <img src="https://user-images.githubusercontent.com/92680829/158732047-c77e2034-a3e3-4fc4-b012-75363800d882.png" width="200" >


- Also Define [nu x 1] matrix θ : information about all users 1 ~ nu
    - <img src="https://user-images.githubusercontent.com/92680829/158732244-8d2e259b-1670-4b8e-9035-2ffc57a8d26e.png" width="250" >


- Now with the matrices θ and  X that we defined above,
    - X * θT equals to the matrix of predicted ratings 
    - This is a **vectorized way of computing the prediction range matrix by doing X * θT**
    - the name of this algorithm is **"Low Rank Matrix Factorization"**
        - low-rank approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data, x) and an approximating matrix (the optimization variable, θ), subject to a constraint that the approximating matrix has reduced rank.
    
    <img src="https://user-images.githubusercontent.com/92680829/158735251-f7e79210-77b6-4718-a01a-0b7fb3df47bc.png"  width="500">


### **Finding Related Movies**
- For each movie (i), we learn x(i) [nx1 vector]
    - x1(i) = romance, x2(i) = comedy, ... xn(i) = action
- Then, how can we find movie j that is closely related to movie i
    - [[xi - xj]]  : can be a good estimate of movie similarity
    - Provides a good indicator of how similar two films are in the sense of user perception

---

<br/>

## **Mean Normalization : implementational detail**

<br/>

- Suppose there is a user 5 that has rated no moives at all
    - <img src="https://user-images.githubusercontent.com/92680829/158736957-1f303b3b-88a5-43a4-b4f6-6fb5f7be41df.png" width="650" >
    
    - then r(i, 5) is always 0 so, the error part (predict - actual) is irrelevant with the cost
    - therefore our goal is to minimize the squared θ5 
    - but, this means that we have to set θ5 = [0, 0], which makes the predicted ratings for all movies is also 0
    - this doesn't make sense!!
    - THEN, how can we solve this problem?

<br/>

### **Mean Normalization**

<br/>

- Group all our ratings into matrix Y [5x4] (5 movie ratings from 4 users)
- Compute the average rating of each movie and store them to a new matrix [nm x 1]
- Subtract off the mean rating from Y 
    - <img src="https://user-images.githubusercontent.com/92680829/158737754-c951a479-464a-42f5-879c-13b6fb9a48a6.png" width="700" >

- Now, let's pretend that newly normalized Y' matrix as our actual data 
- use this Y' to learn θj and Xi 
    - for user j, predicted rating for movie i : (θj)T * (xi) + **(μi)**
    - Now then, we can set all predicted ratings of the empty value as the average (μi) ratings of movie i  
- Back to our previous problem θ5, now we can reset predicted rating of movie i from user 5 
    - (θ5)T xi + μi
    - Where (θ5)T xi = to 0 (still)
    - But we then add the mean (μi) which means user 5 has an average rating assigned to each movie for here
- This makes sense
    - If Eve hasn't rated any films, predict the average rating of the films based on everyone
    - This is the best we can do

- As an aside - we spoke here about mean normalization for users with no ratings
    - But! If you have some **movies with no ratings**, you can also play with versions of the algorithm where you normalize the **columns**
    - BUT this is probably less relevant - probably shouldn't recommend an unrated movie
- To summarize, this shows how to use mean normalization preprocessing to allow your system to deal with users who have not yet made any ratings
    - --> Means system **recommend the best average rated products to the user we know little about**
