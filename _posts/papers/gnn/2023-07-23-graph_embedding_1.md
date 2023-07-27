---
layout: post
title : "[Paper Review & Partial Implementation] Graph Embeddings - Part 1 : DeepWalk, Node2Vec"
img: papers/gnn/gembd1.png
categories: [papers-gnn]  
tag : [Paper Review, GNN, Graph Embedding, DeepWalk, Node2Vec]
toc : true2020
toc_sticky : true
---

## **Outlines**

- [**Reference**](#reference)
- [**Graph Embedding**](#graph-embedding)
- [**1. DeepWalk**](#1-deepwalk)
    - [**1.1. Algorithms for DeepWalk**](#11-algorithms-for-deepwalk)
    - [**1.2. PyTorch Implementation for DeepWalk**](#12-pytorch-implementation-for-deepwalk)    
        - [**1.2.1. RandomWalk**](#121-randomwalk)  
        - [**1.2.2. Hierarchical Skip-Gram**](#122-hierarchical-skip-gram)    
- [**2. Node2Vec**](#2-node2vec)
    - [**2.1. Random Walk with Search Bias**](#21-random-walk-with-search-bias)
    - [**2.2. Comparison to Other Node Embedding Strategies**](#22-comparison-to-other-node-embedding-strategies) 

<br/>

## **Reference**

<br/>

- [**DeepWalk: Online Learning of Social Representations, Perozzi et al, 2014**](https://arxiv.org/pdf/1403.6652.pdf){:target="_blank"}
- [**node2vec: Scalable Feature Learning for Networks, Grover et al, 2017**](https://arxiv.org/pdf/1609.02907.pdf){:target="_blank"}
- [**Comparative Analysis of Unsupervised Protein Similarity Prediction Based on Graph Embedding, Zhang1 et al, 2021**](https://www.researchgate.net/publication/354760987_Comparative_Analysis_of_Unsupervised_Protein_Similarity_Prediction_Based_on_Graph_Embedding){:target="_blank"}
- [**[DL] Hierarchical Softmax**](https://jordano-jackson.tistory.com/72){:target="_blank"}
- [**09-02 워드투벡터(Word2Vec)**](https://wikidocs.net/22660){:target="_blank"}
- [**PyTorch Implementation for DeepWalk**](https://github.com/dsgiitr/graph_nets/tree/master/DeepWalk){:target="_blank"}

<br/>

## **Graph Embedding**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/35695af4-8531-4659-b3b3-b6aac159a1eb" width="530">

- The goal of graph embedding is to transform the nodes and edges of a graph into a lower-dimensional vector space while preserving important structural properties and relationships between nodes. 

- In other words, it is a technique used to represent graph data in a continuous, numerical form suitable for machine learning algorithms.

- There are various techniques for graph embedding, including random walks-based methods, spectral-based methods, and deep learning-based methods like Graph Convolutional Networks (GCNs) and GraphSAGE. 

- For this post, I will review the random walks-based low-dimensional embedding strategies 

<br/>

## **1. DeepWalk**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/6b1b824b-d2f3-496c-8a88-72a32ad7f328" width="850">

- DeepWalk, proposed in 2014, tries to generalize word embedding strategies used in natural languages for learning latent low-dimensional representation of graph networks. 

    - This analogy is supported by the similar **power-law distribution** between vertices in graph and word occurrence in natural language.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/296c43f0-2177-4214-ae22-68e8d1e1091f" width="600">

    - If the degree distribution of a connected graph follows a power law (i.e. scale-free), the frequency which vertices appear in the short random walks will also follow a power-law distribution. 
    
    - Word frequency in natural language follows a similar distribution, and hence, techniques from language modeling account for this type of distributional behavior.

- The DeepWalk method is divided into two parts: **random walk** to obtain node sequences and to **generate node embedding** using **hierarchical skip-gram** analogy.

- It treats the paths created from random walks as "sentences" and applies the Word2Vec algorithm (Skip-Gram) to learn embedded node representations in a continuous vector space.

- Additionally, it adopts **hierarchical softmax** for training word embeddings as an alternative to standard softmax based approaches, whici significanlty improves the computational efficiency.


<br/>

### **1.1. Algorithms for DeepWalk**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/ec4158ce-973e-45cd-b24a-b7cfd4ffa987" width="500">

&emsp;  <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/c26ac44d-4361-4be4-aa12-fd3f9636d398" width="500">

- **Overview of the deepwalk approach**

    - Consists of largely two parts : 1. RandomWalk Generator, 2. Hierarchical Skip-Gram Update 

    - Hyper-parameters are denoted as $d$ for embedding dimension, $\gamma$ for the number of random walks per vertice, $t$ for walk length, and for skip-gram, window size as $w$.

    - Empirically through parameter sensitiviey study, works great on $d = 64, \gamma = 30, w = 10$

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/19dbe1b0-b692-4c2b-a3a5-00ce0da07927" width="750">

<br/>

### **1.2. PyTorch Implementation for DeepWalk**

<br/>

- Graph and hyperparameters

```python
adj_list = [[1,2,3], [0,2,3], [0, 1, 3], [0, 1, 2], [5, 6], [4,6], [4, 5], [1, 3]]
size_vertex = len(adj_list)  # number of vertices

w  = 10            # window size
d  = 64            # embedding size
r  = 30          # walks per vertex
t  = 6            # walk length 
lr = 0.025       # learning rate

v=[0,1,2,3,4,5,6,7] #labels of available vertices
```

<br/>

- Two part : 1. RandomWalk → 2. Learn node embeddings through hierarchical skip-gram. 

```python
for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi = RandomWalk(vi,t)
        HierarchicalSkipGram(wvi, w)
```

<br/>

#### **1.2.1. RandomWalk**

<br/>

```python
def RandomWalk(node,t):
    walk = [node]        # Walk starts from the source node
    
    for i in range(t-1):
        node = adj_list[node][random.randint(0,len(adj_list[node])-1)]
        walk.append(node)

    return walk
```

<br/>

- Randomly selects the next-step node among the adjacent neighboirs connected from the source node and adds to random walk path.

<br/>

#### **1.2.2. Hierarchical Skip-Gram**

<br/>

##### **1.2.2.1 Standard Softmax Based Embedding Learning**

<br/>

- **CBOW (Continuous Bag of Words)**

    - Predict embedding information of the target word from the context words within the window size (-w to +w from the center).

    - Need to compute dot product with V x M lookup matrix (W) for 2w times for all adjacent words.

    - Average out all the resultant vectors to generate embeddings (lookup, $v$) for the central word.

    - Project the embedded vector with M x V projection matrix (W') and take the softmax. 

    - objective function uses cross entropy. 

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/91f71e80-709a-4a03-acc2-d8a672424408" width="700">


- **Skip-Gram**

    - In contrast to CBOW, the goal is to predict context words given a target word.
    
    - Use an embedded vector of the target word to project its context words within the window size.

    - No need to separately compute embedded vectors for all neighboring words and average them to get embedding vector for the target word. 

    - Generate only one embedded vector for the central node using lookup matrix and train projection matrix W' for 2w times, each for its neighboring node. 

    &emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/b19c3871-33f5-40c1-a39b-69f3695a7081" width="600">


- Both CBOW and skip-gram uses softmax function to calculate the probability distribution over the entire vocabulary for each context word, which can be computationally expensive for large vocabularies. 
    
    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/52f70d19-c53a-4f42-8eed-7ea26a1f32e9" width="300">

- Complexity of softmax grows linearly with the vocabulary size, making it slow and memory-intensive to compute probabilities for every word.


<br/>

##### **1.2.2.2 Hierarchical Softmax**

<br/>

- Hierarchical softmax offers a more efficient alternative. Instead of computing the softmax over the entire vocabulary, it creates a binary tree (Huffman Tree) to hierarchically represent the words in the vocabulary. 

- Each word is associated with a unique path from the root of the tree (context word) to its corresponding leaf node (target word). 

- **The probability of a word** in the hierarchical softmax is calculated as the **product of the conditional probabilities along the path to the target word's leaf node**.

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/332a162b-e8af-4626-90b0-632fb3e09dcb" width="650">

- **Conditional Probablity : $\large p(w_{T}\|w_{C})$**

    - Take sigmoid to the dot product of embedding vector of target word (leaf node) and context word. 

    - Multiplier : -1 if right child, +1 if left child so that both sides add up to 1. 

    - **Example**

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/206d5d09-7e29-4900-8716-ce258a0a3765" width="565">


##### **1.2.2.3 Hierarchical Skip-Gram for DeepWalk**

<br/>

```python
# count The length of path from the root node to the given vertex.
def func_L(w):
    count=1
    while(w!=1):
        count+=1
        w//=2

    return count


# returns the nth node in the path from the root node to the given vertex
def func_n(w, j):
    li=[w]
    while(w!=1):
        w = w//2
        li.append(w)

    li.reverse()
    
    return li[j]
```

<br/>

```python
class HierarchicalModel(torch.nn.Module):
    
    def __init__(self):
        super(HierarchicalModel, self).__init__()
        self.phi         = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))   
        self.prob_tensor = nn.Parameter(torch.rand((2*size_vertex, d), requires_grad=True))
    
    def forward(self, wi, wk):
        one_hot     = torch.zeros(size_vertex)
        one_hot[wi] = 1
        wk = size_vertex + wk
        h = torch.matmul(one_hot, self.phi)
        p = torch.tensor([1.0])
        for i in range(1, func_L(wk)-1):
            mult = -1
            if(func_n(wk, i+1)==2*func_n(wk, i)): # Left child
                mult = 1
        
            p = p*sigmoid(mult*torch.matmul(self.prob_tensor[func_n(wk,i)], h))
        
        return p


hierarchicalModel = HierarchicalModel()
```

<br/>

<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/2ad6856f-833f-482f-9c42-4fdcabb00e37" width="660">

<br/>

- Assigning the vertices to the leaves of a binary tree, calculate the probability of a specific path in the tree that connects the root node ($\large \Phi(v_{j})$) to the target leaf node ($\large v_{3}, \, v_{5}$).

- $\large P(u_{k}\, \|\, \Phi(v_{j}))$ :  Given the representation of $\large v_{j}$ (source node in the randomwalk), the probability of a path from the root node to the target leaf node (one of its neighbors in the randomwalk).

    - <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/27068b33-9748-4e1f-975e-4dc2ebb5cf67" width="330">

    - $\large b_{l} = b_{0}, b_{1}, ... , b_{log\|V\|}$ where $\large b_{0}$ : root node, $\large b_{log\|V\|}$ : $\large u_{k}$

    - Hierarchical softmax reduces the computational complexity for calculating $\large P(u_{k}\, \|\, \Phi(v_{j}))$ from $\large O(V)$ to $\large O(logV)$

- $\large w$ : position of $\large u_{k}$. 

- $\large h$ : embedded vector for the root of the tree $\large v_{j}$. 

- func_L(wk) : path length from the root node to the target leaf node.

- func_n(wk, i) : ith node in the path.

<br/>

```python
def HierarchicalSkipGram(wvi,  w):
   
    for j in range(len(wvi)):
        for k in range(max(0,j-w) , min(j+w, len(wvi))):
            prob = hierarchicalModel(wvi[j], wvi[k])
            loss = - torch.log(prob)
            loss.backward()
            for param in hierarchicalModel.parameters():
                param.data.sub_(lr*param.grad)
                param.grad.data.zero_()
```

<br/>

- Visiting all nodes in a given randomwalk path from the starting node to the end node, it tries to learn embeddings that can maximizes the probability of its neighbors (within the window size) in each walk step. 

<br/>
<br/>


## **2. Node2Vec**

<br/>

- node2vec is an extension of DeepWalk, also fundamentally based on the random walk algorithm.

- The key idea is to add the **search bias term $\alpha$** to edge weights that numerically **balances between breadth-first search (BFS) and a depth-first search (DFS) during random walks**. 

    <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/a84fbe97-d523-44dc-9c13-cdc0f073ac60" width="530">

    - **BFS** 
        
        - Constrain the neighborhoods to nodes that are immediate neighbors of the source.

        - Captures micro-view of local connectivity centering around the source node.

    - **DFS**
    
        - Neighborhood consists of nodes sequentially sampled at increasing distances from the source node.

        - Explores broader parts of the network as it moves further away from the source node, and hence, reflects the macro-view of the neighborhoods. 

- Flexibly interpolate between BFS and DFS, node2vec can capture both local and global graph structures, making it highly effective for a wide range of graph-related tasks.


<br/>

### **2.1. Random Walk with Search Bias**

<br/>

#### **2.1.1. Random Walk Probability**

<br/>

<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/91052c4f-b816-4f8a-8c8c-4ff77ea5b650" width="500">

- Starting from a source node $\large u$, a random walk path of length $\large l$

- $\large c_{i}$ : ith node in the walk. ($\large c_{0} \, = \, u$)

- $\large \pi_{vx}$ : unoramlized transition probability from $\large (i-1)^{th}$ node ($\large v$) to $\large i^{th}$ node ($\large x$)

- $\large Z$ : normalizing constant 

<br/>

#### **2.1.2. Determining $\large \pi_{vx}$ with Bias $\alpha$** 

<br/>

&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/687668c8-17a8-4f33-a0b1-d14299d571f0" width="380">

- Considering a random walk that just traversed the edge $\large (t, v)$, and currently reside on node $\large v$.

- Transition probability of edge $\large (v, x)$ is represented as follows

    - $\large \pi_{vx} = \alpha_{pq}(t, x) \, \times \, w_{vx}$  

    &emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d150cdb8-da53-417b-b09e-dc54f11ff98c" width="340">

- $\large d_{tx}$ : shortest path distance from node $\large t$ to $\large x$, one of {$0, 1, 2$}

    - 0 : returns back to $\large t$ 

    - 1 : $\large x$, a common neighbor of both $\large t$ and $\large v$

    - 2 : one step further away from node $\large t$

<br/>

#### **2.1.3. Parameters $\large p$ and $\large q$**

<br/>

- Intuitively, parameters p and q control how fast the walk explores and leaves the neighborhood of source node u.

- In other words, these are the factors for balancing between the two extreme search paradigms, BFS and DFS.

- $\large p$ : return parameter

    - likelihood of immediately revisiting previous node in the walk.

    - $\large d_{tx} \, = \, 0$

    - smaller $\large p$ more likely to lead the walk to backtracking step. 

    - BFS-like exploration.

- $\large q$ : In-Out parameter
        
    - allows the search to distinguish inward and outward propagation. 

    -  $\large d_{tx} \, = \, 2$

    - higher q biases the walk to the nodes close to node $\large t$, reluctant to leaving outwards.

    - lower q, on the other hand, encourages outward exploration, more likely to obtain extended view of the neighborhoods in the walk. 

    - DFS-like exploration.

- setting all p and q to be 1 equals to previously introduced DeepWalk.

- By biasing the random walk probability to flexibly explore various types of network structures, incorporating both BFS and DFS methods, node2vec can learn node embeddings that captures more comprehensive representation of the graph's neighborhood structure.

<br/>

### **2.2. Comparison to Other Node Embedding Strategies**

<br/>

- Efficacy of node2vec over existing state-ofthe-art techniques on multi-label classification and link prediction
in several real-world networks from diverse domains.

&emsp; **Table 2 : Macro-F1 scores for multilabel classification tasks with 50% of the nodes labeled for training.**

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/a251c2e4-80ac-46b1-a4df-8e773705e18e" width="600">

- BlogCatalog: a network of social relationships of the bloggers listed on the BlogCatalog website.

- Protein-Protein Interactions (PPI) : a subgraph of the PPI network for Homo Sapiens.

- Wikipedia : cooccurrence network of words appearing in the first million bytes of the Wikipedia dump.

<br/>

---

<br/>

- Up until now, I've examined two graph embedding methods based on random walks: DeepWalk and node2vec, which are considered relatively early in the field.

- The next post will adress two different graph embedding approaches, SDNE and Graph2Vec, which do not rely on random walk algorithms.