---
layout: post
title : "[Paper Review] Summaries of Papers on Graph Rewiring to Address Over-Squashing"
img: papers/gnn/rewirings.png
categories: [papers-gnn]  
tag : [Paper Review, GNN, Rewiring, Over-Squashing, LRI]
toc : true2020
toc_sticky : true
---

<br/>

Note that all summaries here are for personal purpose to better understand mathematical details of essential Theorems and Lemmas for each rewiring approach and thus may not be appropriate for others who want to get the general glimpse or overview of the papers. 


<br/>

## **Graph Rewiring Strategies**

<br/>

### **1. GDC**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/46d1cbfb-17ce-42df-b643-b4075e44c4aa" width="700">

<br/>

- **Paper** : [**Diffusion Improves Graph Learning, Gasteiger et al, 2019**](https://arxiv.org/abs/1911.05485){:target='_blank'} 

- **Summary Note** : [**2019, Gasteiger et al, GDC (DIGL)**](https://drive.google.com/file/d/1Gar2aGfo5i-ExJdduQ7_RNQgASm7YLNd/view?usp=drive_link){:target='_blank'} 

- **Description** : 

    - Shows direct correspondence between graph diffusion based message passing scheme and spectral graph covolution with the particular choices of diffusion coefficients equivalent to a polynomial filter.

    - Suggest a model GDC (Graph Diffusion Convolution), which is a spectral graph diffusion that combines the strengths of both spatial (message passing) and spectral methods.  

<br/>

### **2. BLEND**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/c69282f2-6f66-4c56-a08d-0ee7bc3bf2f3" width="780">

<br/>

- **Paper** : [**Beltrami Flow and Neural Diffusion on Graphs, Chamberlain et al, 2021**](https://arxiv.org/abs/2110.09443){:target='_blank'}

- **Summary Note** : [**2021, chamberlain, BLEND**](https://drive.google.com/file/d/1jGLE753hiUlMSm5ko2PDI0reXXFspELU/view?usp=drive_link){:target='_blank'} 

- **Description** : 

    - Adopt the concept of discretised Beltrami flow on graphs to model a non-Euclidean graph diffusion PDE that evolves the graph embeddings in joint space (feature and positional coordinates).


<br/>

### **3. FoSR**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d53ee96a-f715-427f-84a0-13cab95f25ad" width="480">

<br/>

- **Paper** : [**FoSR: First-order spectral rewiring for addressing oversquashing in GNNs, Karhadkar et al, 2022**](https://arxiv.org/abs/2210.11790){:target='_blank'} 

- **Summary Note** : [**2023, Karhadkar, FoSR**](https://drive.google.com/file/d/1i3MFBzf4O56RtUKK_9XMLJJdbJZc4ucZ/view?usp=drive_link){:target='_blank'} 

- **Description** : 

    - Base architecture is relational GIN that uses separate aggregation functions for original graph and rewired one to preserve graph topology while easing the information flow across the graph.

    - Use spectral gap (first non-zero eigenvalue of graph laplacian) as a measure of graph connectivity (the ease of graph flow) and add an edge that minimizes the change in spectral gap of normalized adjacency matrix ($\large L \, = \, I - A$). 

<br/>

### **4. GTR**

<br/>

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/5f105b53-2cf1-4fbb-b38b-9456a7cb7550" width="580">

<br/>

- **Paper** : [**Understanding Oversquashing in GNNs through the Lens of Effective Resistance, Black et al, 2022**](https://arxiv.org/abs/2302.06835){:target='_blank'} 

- **Summary Note** : [**2023, black, ER**](https://drive.google.com/file/d/1aUAaPtveDd_wWUzpTm4TAobtz4WA1nSH/view?usp=drive_link){:target='_blank'} 

- **Description** : 

    - Define total effective resistance by summing up the local effective resistances, which captures local bottleneck between two nodes and use it as an approximation of the total amount of oversquashing in a graph. (greedily add edges that maximize the change in R)

    - Provides the connection (upper bound) between the Jacobian of the graph embeddings with arbitrary size of receptive field and the total effective resistance. 

<br/>

---

<br/>

- First two papers are focusing on local (spatial) properties (positional encoding defined on nodes) as an optimization target for graph rewiring algorithms. (not included here, but curvature based rewiring (**SDRF, Topping et al, 2022**) also is a spatial rewiring as the curvature is locally defined on each edge)

- On the other hand, last two papers (**FoSR** and **GTR**) gives more attention to spectral rewiring (spectral gap and total effective resistance) that can capture more global and comprehensive characteristics of graphs.

<br/>
    
&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/1f7a8e12-8075-402a-acea-d58b71efd8af" width="900">

<br/>

- [**Slides**](https://docs.google.com/presentation/d/1adgOB2I-W8Es1N2cqhN-gupnJtbvVlaG/edit?usp=sharing&ouid=106095912423047104850&rtpof=true&sd=true){:target='_blank'} 
