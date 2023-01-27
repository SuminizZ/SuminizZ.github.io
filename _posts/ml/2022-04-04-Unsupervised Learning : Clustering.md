---
layout : post
title : "[Coursera : ML Specialization] - Unsupervised Learning : Clustering"
date: 2022-04-06 00:00:00
# img: autodrive/ose/kalman_filter.jpg
categories: [deeplearning-ml]
tag : [Coursera, ML, Machine Learning]
toc : true
toc_sticky : true
---


<br/>

## **Unsupervised Learning**
<br/>

- training set w/o any label on it
- you are given **an unlabeled dataset** and just ask algorithm to **find the structure of data** 
    - and make it clustered : Clustering Algorithm
- Applications
    - Market segmentation - group customers into different market segments
    - Social network analysis - Facebook "smartlists"
    - Organizing computer clusters and data centers for network layout and location
    - Astronomical data analysis - Understanding galaxy formation

<br/>


## **K-Means Algorithm**
<br/>

- Repeat)
    - Step1. Randomly allocate several points as the cluster centroids
        - you can have as many cluster centroids as clusters you want to do (K cluster centroids, in fact)
    - Step2. Cluster assignment step
        - for every each data point, assign them to clusters depending on the proximity with each cluster centroids
        - and color them with the color of the centroids that they are close to

    <img src="https://user-images.githubusercontent.com/92680829/157780154-11cb254b-ad0b-4f58-b546-90da8117aab5.png" width="700" >
<br/>

 - Step3. Move centroid step
     - move them to the average location of all points colored with the same colour
     
     <img src="https://user-images.githubusercontent.com/92680829/157780644-9a174167-4ee5-4173-8c94-ab68b3a8ad65.png" width="600">
<br/>

- if you keep running additional iterations,
    - when the location of the centroids equals to th avg of clustered data 
    - **Convergence** : end iteration

<br/>

### **Summary of K-means Algorithm**
<br/>

- Input: 
    - K (number of clusters in the data)
        - how to choose K will be covered later
    - Training set {x1, x2, x3 ..., xn) 

- Algorithm:
    1. Randomly initialize K cluster centroids as {μ1, μ2, μ3 ... μK}
    2. for i from 1 to m (m = size of dataset)
        - c(i) : from 1 to K, **index** of the centroid that is closet to x(i)
    3. for k from 1 to K (K = size of centroids)
        - μ(k) : average of data points assigned to kth centorid cluster = location of the centorids (moving centroids step)
        - μc(i) : centroid cluster with the index of c(i), the cluster to which x(i) has been assigned
    4. Repeat the loop 
        - Loop1 : 
            - repeatedly sets the c(i) variable to be the index of the variable of cluster centroid closest to xi 
            - i.e. take ith example, measure squared distance to each cluster centroid, assign c(i) to the cluster closest
                - min [ x(i) - μ(k) ]^2
        - Loop2 : 
            - depending on the results of the first loop, move all centroids to the average of newly assigned data points 
            - = reset the μ(k) for k in 1 to K

- What if there's a centroid with no data
    - Remove that centroid, so end up with K-1 classes

<br/>

### **K-means for non-separated clusters**
<br/>

- often K-means is applied to datasets where there aren't well defined clusters
    - e.g. T-shirt sizing : 
       - k-means can even cluster the data that doesn't look to have any clear clusters
       - example of market segmentation where you use k-means to separate your market and design the products depending on it
       - <img src="https://user-images.githubusercontent.com/92680829/157783656-5507974b-7a8a-4334-a6e5-64ece66950be.png" width="300" >

---
<br/>

## **Optimization Objectives of K-means**
- cost function that k-means algorithm tries to minimize
- that is the sum of sqaured distance between x(i) and μc(i)
    - <img src="https://user-images.githubusercontent.com/92680829/157784513-81e710a2-2771-4a51-8dbe-286c8b9e1278.png" width="500">

<br/>

- optmization objective : min J(c,μ)
    - The **cluster assignment step** is minimizing J(...) with respect to c1, c2 ... ci 
        - i.e. find the centroid (c(i)) closest to each example (x(i))
        - Doesn't change the centroids themselves
    - The move centroid step
        - choosing the values of μ which minimizes J(...) with respect to μc(i)..
        - this step actually changes the cluster centroids
        


- So, we're partitioning the algorithm into two parts
    - First part minimizes the c variables
    - Second part minimizes the J variables
- We can use this knowledge to help **debug our K-means algorithm**

---
<br/>

## **Random Initialization : Debugging**
- how k-means can avoid converging to **Local Optima**
    - Randomize initial K
        - should have K < m (m = size of x)
        - randomly pick K from training examples (i(1)... i(m))
        - set μ1 ... μk equal to these K examples 
- K means can converge to different solutions depending on the initialization setup
    - Risk of local optimum
    - <img src="https://user-images.githubusercontent.com/92680829/157786991-b70a266d-da4e-4935-96aa-09b715ef9b83.png" width="500">
    
<br/>

### **Try Multiple Random Initialization**
<br/>

- <img src="https://user-images.githubusercontent.com/92680829/157787440-ffa3c99a-b6ba-4561-b1e9-b062c27e6351.png" width="350" >

- Now you have 100 differenct cases of clustering and cost 
- pick one with lowest cost

<br/>

### **Choosing the number of clusters : K**
<br/>

- most common way : choosing it manually by observing visualization result
- since its not labeled dataset, it's generally quite ambiguous to tell how many clusters are there
- a few of them are presented below

- **1. Elbow Method**
    - plot J by every K size that you've tried
    - <img src="https://user-images.githubusercontent.com/92680829/157788594-25d597eb-18b7-468b-be31-bd58b845944d.png" width="300" >
    - ideally, you can set your k where the "elbow" is formed
    - but practically, usually the location of elbow is not so clear and still seems ambiguous which K is the best clustering size
    

- 2. **Considering your Business Considerations**
    - people usually use k-means for later/downstream purpose in market segmentation
    - not just in terms of minimizing cost, you can consider other issues related to your business or market

