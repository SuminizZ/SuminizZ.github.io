---
layout: post
title : "[Graph Theory] Markov Chains : Part 1 - Basic Properties and Classfications of Markov Chains"
img: ml/spgt/mkv11.png
categories: [ml-spgt]  
tag : [ML, Markov Chains, Graph Theory]
toc : true
toc_sticky : true
---

## **Outlines** 
<br/>

- [**References**](#references)
- [**1. Markov Process**](#1-markov-process)
- [**2. Evolution of Markov Chains**](#2-evolution-of-markov-chains)
    - [**2.1. Stationary Distribution and Eigendecomposition**](#21-stationary-distribution-and-eigendecomposition)
- [**3. Classifications of Markov Chains**](#3-classifications-of-markov-chains)
    - [**3.1. Various Properties of States**](#31-various-properties-of-states)
    - [**3.2. Ergodic Chains**](#32-ergodic-chains)
    - [**3.3. Reversible Chains**](#33-reversible-chains)
    - [**3.4. Absorbing Chains**](#34-absorbing-chains)

<br/>

## **References**

- [**A Tutorial on the Spectral Theory of Markov Chains, Seabrook E, Wiskott L., 2023**](https://arxiv.org/abs/2207.02296){:target="_blank"}
- [**Everything about Markov Chains., Cai Leran**](https://www.cl.cam.ac.uk/~lc647/drafts/Markov_Chains.pdf){:target='_blank'}
- [**MIT OCW 6.041 Probabilistic Systems Analysis And Applied Probability - Lecture 16, John Tsitsiklis**](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/){:target='_blank'}

<br/>

## **1. Markov Process**

<br/>

- Markov chains are widely used tool for modelling stochastic and dynamic processes that temporally evolve on the graphs. 

- As a well-established statistical framework, Markov chain model provides a lot of practical advantages in terms of numerical computations and interpretability when applied to various time-series graphs.  

- The fundamental assumption regarding to the Markov chains is that their evolution is independent of past memory in the long run and only depends on the current states and transition probabilities, which is called as Markov process.

    - Given past states $\large Z$, current states $\large Y$, and future states $\large X$, 
    
    - $\large Pr(X \| Y, Z) \, = \, Pr(X \| Y)$

- A Markov chain refers to a process that consists of a set of states (can be finite or countably infinite) whose evolution is defined on a discretized time domain.

- The rule for state transition in the Markov chain is described in terms of transition probabilities between pairs of states in $\large \mathcal{S}$, a state space.

- When these probabilities are invariant of time, the Markov chain is said to be time homogenous and its temporal evolution can be fully described with the given transition matrix. 

- The transition probabilities lie on a matrix called **transition matrix** $\large P \, \in \, \mathbb{R}^{N \times N}$ where $N$ is the number of possible states. 

- **Example (a)**

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/3bbe74c6-abc8-4921-8760-0bf7cb26e107" width="630">

    - $\large P_{ij}$ indicates the probability of 1-step transition from state $\large i$ to $\large j$ at a given time $\large t$.
    
    - Formally, defined as $\large Pr(X_{t+1}(j) \, \| \, X_{t}(i))$

- Each row of transition matrix $\large P_{i}$ describes the stochastic probabilities of transition into state $\large i$ from each of $\large N$ states, which sum up to 1. 

- On the other hand, coloum vectors of $\large P_{j}$ don't add up to 1.

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/e3687da4-e5d4-4adb-aee2-f5129df46444" width="520">

- The figure above represents the result for $\large n$ repeated simulations of the given Markov chain**.

- Initially starting from the state S, relative state occupations after 1-step transition are colored in gray-scale.  

- As the iteration increases from 10 to 1000, the distribution of states becomes close to 
    
    $\large P_{S} \, = \, Pr(X_{1} \| X_{0} = S) \, = \, (0.5, 0.1, 0.2, 0.2)$
    
<br/>

## **2. Evolution of Markov Chains**

<br/>

- The evolution of Markov chains is done by a multiplication of transition matrix $\large P$ from the right to current state vector at time $\large t$, $\large \mu(t) \, \in \, \mathbb{R}^{N}$. 

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d4341311-d5c7-4c94-9425-5b9a2388ad1b" width="230"> where <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/8b749f75-8710-4d58-bd06-400deea9076f" width="230">

- As mentioned in previous [Section 1.](#1-markov-process), all entries of each row vector of $\large P$ sum to 1.

- Multipying $\large P$ from the right results in a linear combination of each row vector in $\large P$ where $\large P$ to the any power $\large k$ preserves the stochastic distribution, therby the sum over all states stays constant upon multiplying $\large P$.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/e7b9344b-8194-4d9f-9697-a8218b19f5a1" width="470">

- From the recurrent relations between $\large \mu(t+1)$ and $\large \mu(t)$, $\large k$ steps of evolution can be expressed as
    
    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/48c36e09-3ded-495f-9c15-225fcacb79ce" width="350">

<br/>

### **2.1. Stationary Distribution and Eigendecomposition**

<br/>

- For some Markov chains that satisfy certain conditions, which will be discussed with further details in later Section, after a series of evolution, the state distribution of the Markov chains converges to a stationary distribution $\large \pi$ and then stays invariant in further steps, i.e.,

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/22941dea-01a3-4ca8-a035-27125dc34b6c" width="170">

- Intuitively, this means the total flow of probability mass $\large F^{\pi}$ from all states into each state $\large s_{j}$ is constant.

    - Flow matrix $\large F^{\Pi} \,:=\, \Pi P$ where $\large \Pi$ is a diagonal matrix consisting of the entries of $\large \pi$.
    
        - the total flow to $\large s_{j}$ is $\large F_{j}^{\pi} \,=\, \sum_{i=1}^{N} \pi_{j} P_{ij}$ 

- From the equation $\large \pi P \,=\ \pi$, we can see that $\large P$ needs to have at least one left eigenvector with eigenvalue equal to 1.

- Further, it is straightforward to see that $\large \eta \, = \, (1, 1, 1, ..., 1)^{T}$ has to be a right eigenvector cosidiering that row sum of $\large P$ is one.

    - $\large P\eta \, = \, (1, 1, 1, ..., 1)^{T}$  
    &nbsp;

- Hence, Markov chain that has stationary distribution $\large \pi$ always has $\large \pi$ as its left eigenvector and $\large \eta$ as it right eigenvector.

- When the transition matrix $\large P$ is diagonalizable, the eidgendecomposition of $\large P$ is given as

    - $\large P \, = \, Y_{R} \, \Lambda \, Y_{R}^{-1}$ where $\large Y_{R} \,\in\, \mathbb{R}^{N\times N}$ is the matrix whose $\large j$-th column vector is right eigenvector $\large r_{j} \,\in\, \mathbb{R}^{N}$ and $\large Y_{R}^{-1}$ consists of row vectors corresponding to the transpose of left eigenvectors $\large l_{i} \,\in\, \mathbb{R}^{N}$ of $\large P$. 

    - $\large \Lambda$ is a diagonal matrix where each $\large \lambda_{ii}$ is the eigenvalue of $\large i$-th left/right eigenvectors. 
 
    &emsp;&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/8b7336b5-9da4-4f8d-8f3f-6bbfdb17aa54" width="450">

    - Left and right eigenvectors are orthogonal to each other unless they share same eigenvalue. 

        - <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/43ec4b0b-a0f0-4c59-a3e1-ec10215e1228" width="250">

        - if $\large \lambda_{\omega} \, \neq \, \lambda_{\gamma}$, then $\large l_{\omega}^{T}r_{\gamma}$ is zero. 

- A powerful advantage of doing this is that once you can diagonalize $\large P$, matrix multiplication required for the evolution of Markov chains can be transformed to much simpler computation, a scalar multiplication. 

    - Since each eigenvector is orthogonal to others, they form a basis of eigenspace that spans $\large R^{N}$ and thus $ N$-dimensional states $\large \mu(t)$ can be expressed in terms of this eigenbasis with arbitrary coefficients $\large c_{w}$.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/7496a1f5-c223-47bd-b596-123e40620282" width="200">

        Then one step evolution using $\large P$ is as follows ($\large l_{\omega}^{T}r_{\gamma}$ cancelled)

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/bcc1e785-8427-4116-856f-79c1e1942017" width="280">

        Following this, k-step evolution can be earned simply by k-th power of eigenvalues.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/bb951f29-b9fd-489f-91b4-a9493b640a4d" width="280">
    
- Beyond the efficiency of computations, another practical advantage that the spectral decomposition of $\large P$ can provide is that it can tell the fundamental properties of the given Markov chains, such as persistency and periodicity. 

    - Since this part will be convered in [Part 2](){:target="_blank"} in depth, I will briefly mention about how eigenspace of the transition matrix is associated with the behavior of Markov chains.

    - From the popular spectral graph theory, ***Perron-Frobenius theorem***, the spectral radius of eigenvalues of the diagonalizable $\large P$ is 1. (i.e., $\large \|\lambda\| \, \leq \, 1$)

    - Dividing the set of eigenvalues into $\large \|\lambda\| = 1$ and those that are not, the evolution of Markov chains is re-written as 

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/3500ae11-81fb-4ca9-b3de-3bbff8faf523" width="435">

        - Terms with $\large \|\lambda_{\omega}\| < 1$ exponentially decays with time and thus only have transient effect in the early evoluton, while terms with $\large \|\lambda_{\omega}\| = 1$ stays persistent throughtout the whole evolution untill convergence.

        - Accordingly, the magnitude of $\large \|\lambda_{\omega}\|$ can measure the rate of decay of the transient phase.

    - In addition, partitioning $\large \|\lambda\|$ into three types, (i) $\lambda = 1$, (ii) $\lambda = −1$, and (iii) $\lambda \in \mathbb{C}$, $\|\lambda\| = 1$, each of these types also gives the information about the periodic behaviors of the Markov chains.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d8235ce4-104f-4bcc-ab28-4bce372f9b71" width="620">

        - Further explanations about this figure is in [Part 2](){:target="_blank"}.

<br/>

## **3. Classifications of Markov Chains**

<br/>

- This section will review several types of states and classifications of Markov chains in relation to the transition matrix (i.e., connectivity of Markov chains). 

- First, let me introduce an important definition, a **communicating class**, that is a subset of states $\large \mathcal{S}$ where each state contained in the subset practically communicate with each other.

- Communicating class refers to the set of states that satisfy **equivalence relation** between each of them. 

    1. **Reflective** : $\large \forall s_{i} \in \mathcal{S}, \, s_{i} \leftrightarrow s_{i}$. every state can reach to itself in 0 step, i.e., $\forall i\,, \large P_{ii} = 1$.

    2. **Symmetric** : If $\large s_{i} \rightarrow s_{j}$, then $\large s_{i} \leftarrow s_{j}$.

    3. **Transitive** : If $\large s_{i} \leftrightarrow s_{k}$ and $\large s_{k} \leftrightarrow s_{j}$, then $\large s_{i} \leftrightarrow s_{j}$.

<br/>

### **3.1. Various Properties of States**

<br/>

##### **Reducibility**
&nbsp;
- The number of possible communicating classes is denoted as $\large n$, and chain with $\large n$ = 1 is irreducible while chain with $\large n$ greater than 1 is reducible, meaning that the states can be partitioned into several communicating subsets.  
&nbsp;

##### **Recurrence/Transience**
&nbsp;

- A state $\large s_{i}$ is recurrent if,

    Initially starting from $\large s_{i}$, Markov chain returns to the state $\large s_{i}$ in any arbitrary distance $\large k$ with probability 1, i.e., 
    
    &emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/fa769dfd-d3eb-4c0c-9e1c-097ec8c37535" width="420">

- States with $\large f_{i}$ smaller than 1 are called transient states.

- Same notions can be applied to define the properties of communicating class, if all states in a communicating class are recurrent, then the class itself is recurrent, namely **recurrent class**, otherwise, **transient class**.

- Further, extending to the whole chain, a Markov chain is recurrent if all classes in the chain are recurrent.

&nbsp;

###### **Examples of states with different connetivity**

&emsp;&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/59ee7d12-8359-4822-8315-6fea58aca1a9" width="730">

- Seeing (b) in the figure, states 1 and 4 don't satisfy the equivalence relation and thus partitioned into different communicating classes.

- Since entire states in Markov chain (c) can be contained in one communicating class, the chain is irreducible. 

- Irreducible chain is always recurrent while the converse doesn't hold, which is the case in example (a).

&nbsp;
##### **Multiple Stationary Distributions in Reducible Chains**
&nbsp;

- Each class within a reducible chain has its own stationary distribution corresponding to the class.

- Let a Markov chain has $\large r$ recurrent classes and $\large t$ transient classes.

    - Stationary distributions for states belonging to any transient class is zero. (eventually it leaks out to recurrent classes.)

    - Given $k$-th recurrent class, $\large \pi_{k}$, the probabilities for states within the class is non-zero. 

- Then the stationary distribution for the entire chain is given by an arbitrary combination of $\large r$ distributions of recurrent classes.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/83518f63-94ea-4ef0-a231-bb55ee16c227" width="580">

    - Specific choices of $\large \alpha_{k}$ are infinite, thus so are the possible stationary distributions of the reducible chain.

    - If $\large t = 0$, then $\large \pi$ of the chain is strictly positive.
    
- **Indexing of each class**

    - Given a reducible recurrent chain, the transition matrices and stationary distributions of the chain can be indexed with a block diagonal form, where each $\large P_{k}$ and $\large \Pi_{k}$ contain only the probabilities for states in the $\large k$-th class.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/8564ef1e-ff66-470a-a4b1-7ce92b784cb5" width="530">
&nbsp;

##### **Periodicity**
&nbsp;

- Another fundamental properties defined on the states is periodicity, indicating how frequently a state is revisited by a Markov chain with a certain period.

- Formally, a chain is periodic if 

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/557ac300-fadc-4467-a551-112a565c2803" width="310">

    - gcd refers to the *greatest common divisor*

    - Intutitively the period of state $\large s_{i}$, $\large d_{i}$, is the gcd of path lengths $k$ for self-recurrence that gives non-zero probability. 

    - States for which $\large d_{i} > 1$ are called periodic and those with $\large d_{i} = 1$ is aperiodic.

- **Examples of Periodic staes**

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/5f70956e-31ec-41e7-92b6-1e982f220b42" width="650">

    - All staes in example (b) has $\large d \,=\, 3$ and for (c), $\large d \,=\, 2$ and accordingly all states in the graph are periodic.

    - In contrast, all states in (d) are aperiodic with $\large s_{1}$ having a self-loop that guarantees gcd to be 1 and both $\large s_{2}$, $\large s_{3}$ having $k$ 2 and 3, which are prime numbers, and thus gcd as 1.

- The concept of periodicity can be generalized to the chain.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/2ac85e91-b5af-46ce-a4c4-cc88f3b6c9e7" width="830">

    - Example of periodic chain is shown in (b), (c) and (d) for aperiodic chain.

    - Example of chain with mixed periodicity

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/3705606b-22fa-4438-af13-1091572cd919" width="260">

        - State 2 and 3 are transient and only recurrent paths for them is 2 $\large \rightarrow $ 3 $\large \rightarrow $ 2 and 3 $\large \rightarrow $ 2 $\large \rightarrow $ 3, both giving $\large k = 2$.

        - State 1 has self-loop and thus $\large d_{1} = 1$. 

- Using these various properties of states discussed in this section, next section will identify several classifications of Markov chains. 

<br/>

### **3.2. Ergodic Chains**

<br/>

- First class of Markov chaisn is ergodic chains, chains that guarantees the existence of a unique stationary distribution.

- The beauty of Markov chain model is that it enables statistical predictions for the long term behavior of some stochastic processes regardless of the details in the trajectories that lead to that behavior. 

- This is possible because with enough temporal evolution, the process described by Markov chains converges to a unique distribution that does not change afterwards.

- However, there are two necessary conditions that a chain needs to satisfy to achieve the convergence; **aperiodicity** and **irreducibility**.

    - First, aperiodicity is straightforward to understand, since if the distribution osillates with certain period d, then it never converges to a unique stationary distribution. (i.e., repeating sequence of multiple distributions) 

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/150fbc5b-c6a4-4a11-88b6-db9afb09de57" width="450">

    - Irreducibility ensures that every state is reachable from every state with positive probability, not trapped in smaller subsets of the state space.

- Thus, irreducibility and aperiodicity combined, ergodicity is a strong property ensuring that the chain converges to a unique stationary distribution (i.e., limiting distribution) where every state is recurrent. 

- In addition to the existence of a unique stationary distribution $\large \pi$, the transition matrix $\large P$ is diagonalizable which allows us to take advantage of an eigenspace of $\large P$ given by eigendecomposition.

    &emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/8819403b-2ec1-4f09-a7d0-1f0f2dad1d75" width="310">

    - Provides improved computational efficiency and easier interpretation of the persistent and transient behaviors of the process.



<br/>

### **3.3. Reversible Chains**

<br/>

- Given past $\large Z$, present $\large Y$, and future $\large X$ states, Markov assumption gives the main result, $\large Pr(X \| Y, X) \, = \, Pr(X \| Y)$.

- Using this assumption, the reversed probability of $\large Z$ given $\large Y$ and $\large X$ is as,

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/e22ae4f7-f4e5-4ca9-a7ac-e0f646a8c8f9" width="450">

- This symmetry shows that the reverse of Markov process itself is a Markov process

- Relationship between the transition matrix of reverse Markov process $\large P_{rev}$ and $\large P$ is solved as, 
    
    - <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d4c6f1eb-bd29-4315-8a14-02c2dbefa9d0" width="780">

    - In a vectorized representation, <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/124264b5-32d3-4c76-8829-adc403f05e37" width="180">

        - Now the column vectors in $\large P^{T}$ are stochastic, thus multiplied from the left.

    - Given a forward Markov chain $\large \chi$, the reverse Markov chains using this $\large P_{rev}$ is the time-reversal of $\large \chi$, denoted as $\large \tilde{\chi}$.

    - Identical relation can be applied to reducilble recurrent chains with infinite stationary distributions. 

        - $\large P_{rev}$ is independent of specific choices of $\large \alpha_{k}$ corresponding to each distribution $\large \pi_{k}$. 

            <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/cae88cbc-9191-4fec-a516-673675c0472b" width="360">

        - $\large P_{rev, k}$ is the transition matrix of $\large k$-th recurrent class that evolves via time-reversal $\large \tilde{\chi}$.

- Along with stronger reversibility condition, called as **Detailed Balanced**, $\large P_{rev}$ becomes statistically equivalent to $\large P$

    - Detailed Balance condition

        - <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/ce94a4c9-5bdb-417b-b446-9209c85f5b40" width="270">

        - This gives $\large P^{T}\Pi$ = $\large \Pi P$, which then results in an equality $\large P_{rev}$ = $\large P$

    - Then, the flow matrices of forward and backward transition are also the same.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/db8dc438-e92f-4b1f-880c-b10278b7f0b5" width="175">

- Since reversibility significantly improves both analytic and numerical convenience, there are several methods to modify a non-reversible chain into a  reversible chain, which is a process called ***Reversibilization***. 

    - **Additive Reversibilization**

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/852e6552-599f-4136-a027-e3af4e367d06" width="245">

    - **Multiplicative Reversibilization**

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/e82b6929-1db7-497a-a980-a8c9eb5c9114" width="225">

<br/>

### **3.4. Absorbing Chains**

<br/>

- Final classfication of the Markov chains is absorbing chain, which is tightly related to the concept of recurrence and transience.

- Absorbing chains are the chains that contain at least one absorbing state where transition into the state never returns to other states out of it. 

- States other than an absorbing state are non-absorbing states and they're all transient since Markov chains starting from them will eventually get trapped into the absorbing state and can't revisit them. 

- The transition matrix of the absorbing chains can be arranged in a block strucuture, called as ***canonical form***, as follows, 

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/6764051e-4b8d-4fc0-954d-53936c931972" width="215">

    - Given an absorbing Markov chain with $\large r$ absorbing states and $\large t$ transient states, 
        
        - $\large Q \in \mathbb{R}^{t \times t}$ are a block of transition probabilites only between the transition states.

        - $\large R \in \mathbb{R}^{t \times r}$ contains the probabilities of transition from transient states to absorbing states.

        - $\large 0 \in \mathbb{R}^{r \times t}$, a matrix whose entire entries are 0, representing the recurrent possbilities from absorbing states. 

        - $\large \unicode{x1D7D9} \in \mathbb{R}^{r \times r}$, an identity matrix that implies any transition into the absorbing states stays there with no exit. 

- **Example**

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/68ffa5b1-9399-4abd-b173-6fcd0459aa91" width="620">

- **Neumann series of $\large Q$**

    - As a fundamental matrix of the Markov chain, it carries an important information about the interactions between transient states. 
    
    - From the transience property of $\large Q$, we can easily consider that repeated application of $\large Q$ in an infinite time results in zero distribution for the transition states, i.e., $\large \lim_{n\rightarrow \infty} Q^{n} \, = \, 0$

    - Accordingly, an infinite sum of all powers of $\large Q$ will converge to the following matrix, $\large (\unicode{x1D7D9} - Q)^{-1}$.

        &emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/50f9b1c8-ec44-479e-8eb1-8b5d1a485a5e" width="320">

        - Since $\large k$-th power of $\large Q$, $\large Q^{k}$ contains the number of possbile path of length $\large k$ between pairs of transition states, the sum of them returns the number of all possible paths connecting them with any lengths. 

        - Then, $\large N_{ij}$ gives the expected number of times to visit a transition state $\large s_{j}$ from another transition state $\large s_{i}$ before absorption. 

<br/>

--- 

<br/>

- So far, I've introduced the concept of Markov chains and reviewed the basic but fundamental properties of states that determine the particular classifications of the entire cahin.

- **Summary**

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/4ec5aca2-3fa6-4d3a-9094-dd0ae38630cc" width="610">

- Next post on Markov chains will mainly focus on the eigenspace of Markov chain and its significance implications on the dynamic behaviors of the Markov chains.


