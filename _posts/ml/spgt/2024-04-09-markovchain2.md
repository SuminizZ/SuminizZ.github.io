---
layout: post
title : "[Graph Theory] Markov Chains : Part 2 - Spectral View of Markov Chains and Random Walks"
img: ml/spgt/mkv2.png
categories: [ml-spgt]  
tag : [ML, Markov Chains, Graph Theory]
toc : true
toc_sticky : true
---

## **Outlines** 
<br/>

- [**References**](#references)
- [**1. Eigenspace of Transition Matrix**](#1-weak-generalization-power-of-sharp-minima)
- [**2. Spectral Analysis of Markov Chains**](#2-weak-generalization-power-of-sharp-minima)
    - [**2.1. Persistent and Transient Behavior**](#21-pac-bayesian-generalization-bound)
    - [**2.2. Periodic Behavior**](#22-sam-objective)
- [**3. Random Walks on Graphs**](#3-sharpness-aware-minimization-sam)

<br/>

## **References**

- [**A Tutorial on the Spectral Theory of Markov Chains, Seabrook E, Wiskott L., 2023**](https://arxiv.org/abs/2207.02296){:target="_blank"}
- [**Everything about Markov Chains., Cai Leran**](https://www.cl.cam.ac.uk/~lc647/drafts/Markov_Chains.pdf){:target='_blank'}
- [**MIT OCW 6.041 Probabilistic Systems Analysis And Applied Probability - Lecture 16, John Tsitsiklis**](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/){:target='_blank'}

<br/>

## **1. Eigenspace of Transition Matrix**

<br/>

- As previously discussed in [Part 1](https://suminizz.github.io/markovchain1/){:target="_blank"}, the eigenspace of the transition matrix $\large P$ has great significance in interpreting the dynamic behavior of Markov chains.

- Given a **irreducible Markov chain**, $\large P$ of the chain is diagonalized as $\large Y_{R} \, \Lambda \, Y_{R}^{-1}$ where $\large Y_{R} \,\in\, \mathbb{R}^{N\times N}$ is the matrix whose $\large j$-th column vector is right eigenvector $\large r_{j} \,\in\, \mathbb{R}^{N}$ and $\large Y_{R}^{-1}$ consists of row vectors corresponding to the transpose of left eigenvectors $\large l_{i} \,\in\, \mathbb{R}^{N}$ of $\large P$. 

- $\large \Lambda$ is a diagonal matrix whose $\large i$-th entry is the eigenvalue of $\large i$-th pair of left and right eigenvectors. 

&emsp;&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/8b7336b5-9da4-4f8d-8f3f-6bbfdb17aa54" width="480">

- Regarding to the eigenspace of diagonalizable $\large P$, the main statements referred in this section is following three.

    1. **All complex modulus of eigenvalue $\large \|\lambda\| \, \leq \, 1$.** 

        - This is the result from a well-known *Perron-Frobenius theorem*, saying that for a non-negative irreducible matrix, there exists a unique positive eigenvalue (the Perron root, $\large \lambda_{pf}$) corresponding to a positive eigenvector, and it's the largest eigenvalue in modulus.
            
            - Then, $\large \lambda_{pf}$ is the spectral radius of the matrix.
        
            - Even though the theorem directly limits the matrix to be irreducible, can generalize it to smaller irreducible blocks within the reducible matrices.

        - $\large \lambda = 1$ is the dominant eigenvalue (i.e., $\large \lambda_{pf}$) and thus any other eigenvalues are smaller than 1.
        
        - Intuitively, consider any $\large \lambda$ larger than 1, then the Markov chain, described as $\large \sigma P^{t} \, = \, \pi \, + \, \sum_{i=1}^{N} c_{i} \lambda_{i}^{t} l_{i}^{T}$, will exponentially grow with time or otherwise $\large \sigma P^{t}$ contains some negative entries as t grows, both contradicting the fact that $\large P$ is a stochastic probability distribution. 

    2. **The eigenspace of $\large \lambda = 1$ is guaranteed to be one-dimensional in irreducible chain.**

        - $\large \pi P = \pi$, which makes $\large \pi$ a left-eigenvector with eigenvalue 1. 

        - $\large \pi$ is not necessarily a stationary distribution of the chain.

        - In this case, the spectral radius of $\large P$ is 1.

    3. **For irreducible and aperiodic chains, $\large \|\lambda\| = 1$ is only possible for $\large \lambda = 1$.**
        
<br/>

## **2. Spectral Analysis of Markov Chains**

<br/>

- By analyzing the spectral distribution of eigenvalues of $\large P$, we can get some fundamental information about the Markov chains.

- Firstly, I will consider the case of irreducible chains that have $\large \lambda$ 1 and then generalize the results to reducible chains. 

- First of all, the temporal evolution of a Markov chain is fully described by a series of powers of $\large \lambda$.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/bb951f29-b9fd-489f-91b4-a9493b640a4d" width="280">




































