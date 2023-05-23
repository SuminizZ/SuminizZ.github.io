---
layout: post
title : "[Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau & Bengio (2015)"
img: papers/attention_2015.png
categories: [papers-transformer]  
tag : [Paper Review, Attention, Transformer]
toc : true
toc_sticky : true
---

### **# Reference**
- Bahdanau, Cho, & Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015.
- [Attention for RNN Seq2Seq Models](https://www.youtube.com/watch?v=B3uws4cLcFw&list=PLgtf4d9zHHO8p_zDKstvqvtkv80jhHxoE&index=1){:target="_blank"}

<br/>

### **# Attention for RNN Encoder-Decoder Networks**

&emsp;This paper proposes a novel approach called "Attention" to improve the performance of machine translation using encoder-decoder (Seq2Seq) architeture. 

Encoder-decoders refers to a system where the model encodes a source sentence into a fixed-length vector from which the decoder outputs a translation corresponding to the given source sentence. 

Basic encoder-decoder network has limited performance on the translation of long sentences and this paper successfully mitigates the issue by introducing the concept of "attention" that allows the model to automatically focus on the information relevant to the predicting target word.  

<p align="center"><img src="https://github.com/SuminizZ/Physics/assets/92680829/d61fe383-84ed-4ce6-ac64-ae11d354abda" width="700px"></p>


<br/>

### **# Issue of Interest**

&emsp;The underlying cause behind the poor performance of original encoder-decoder network mainly lies on the fact that the encoder needs to compress the source sentence, regardless of its original legnth, into a fixed-length vector.

The encoder takes a variable-length input and transforms it into a state with a fixed shape and the decoder maps the encoded fixed shaped vector into again, variable-length translated output.

Use of this limited length of context vector acts as an information bottleneck in a sense that as the length of the source sentences increases, more information needs to be squashed and packed into that fixed length context vector, which results in the loss of detailed or possibly important information of the original source input.

This can be shown in the Figure 2. presented above where the BLEU score of the model with basic encoder tends to decrease as the length of source sentence increases.

<br/>

### **# Model Architectures of BiRNN + Attention**

&emsp;The most common encoder-decoder framework used in machine translation is RNN. Here, This is the detailed architecture of proposed attention RNN model (RNNsearch) used in the paper.

<p align="center"><img src="https://github.com/SuminizZ/Physics/assets/92680829/75d63a2f-3df2-46cb-9048-116ee8ca4ffc" width="800px"></p>

(image from [https://www.youtube.com/watch?v=S2msiG9g7Us](https://www.youtube.com/watch?v=S2msiG9g7Us){:target="_blank"} )

<br/>


#### **1. Encoder**

&emsp;First, the model takes source sentences at each time step as input, and compute the forward and backward states of them. 


- **Input (Source Sentence) & Output (Translation) :** <br/>

&emsp;&emsp;&emsp;$\large x = (x_1, \ldots, x_{T_x}), \quad x_i \in \mathbb{R}^{K_x}$  <br/>
&emsp;&emsp;&emsp;$\large y = (y_1, \ldots, y_{T_y}), \quad y_i \in \mathbb{R}^{K_y}$ <br/> 

&emsp;&emsp;&emsp;$T_{x}$ and $T_{y}$ respectively denote the lengths of source and target sentences.

- **Bidirectional RNN (BiRNN) Model :** 

    <img src="https://github.com/SuminizZ/Physics/assets/92680829/8fcfe692-9c59-4aab-9a63-4dbd73ed8b8c" width="400px"> <br/>

    $\large E \in \mathbb{R}^{m \times K_x}$ : word embedding matrix  <br/>

    $\large W, \vec{W}_z, \vec{W}_r \in \mathbb{R}^{n \times m}$ : weight matrices where m denotes the embedding dimensionality <br/>

    $\large \vec{U}, \vec{U}_z, \vec{U}_r \in \mathbb{R}^{n \times n}$ : weight matrices where n denotes the number of hidden units <br/>

    Repeat the same step backwards to get backward states of input. (embedding matrix is shared unlike the weight matrices) <br/> 
    vertically Concatenate the forward and backward states into one complete hidden states matrix.

    $\large h_i = \begin{bmatrix} \overrightarrow{h}_i  & \\
                  \overleftarrow{h}_i \end{bmatrix}^\intercal$  

<Br/>

#### **2. Decoder** 

- <p align="left"><img src="https://github.com/SuminizZ/Physics/assets/92680829/96441f0c-8892-4af0-bb03-ca7a5ff60269" width="260px"></p>

- **Alignment Model** (Different from what's in the paper, but more generally used)

    - $\large k_{i}\,=\, W_{K} \times h_{i}$   (for i = 1 to m, m : number of hidden states in encoder)
    
    - $\large q_{j}\,=\, W_{Q} \times s_{j}$ 

    - Take inner product with $k_{i}$ and $q_{j}$ and normalize it so that $\alpha_{i}$ adds up to 1<br/> 
        - Search for a set of positions ($i$) in a source sentence that is most relevant to hidden state($s_{j}$) of current predicting word.
        - $\large \alpha_{i} = Softmax(k_{i}.dot(q_{j}))$  (for i = 1 to m)
        - $\large \alpha_{i}$ represents how much each hidden state of source sentence contributes to predicting the translation in the decoder.<br/> 

    - $\large \alpha_{i}$ = align($h_{i}$, $s_{j}$) = $\large \frac{\exp(k_i^\top q)}{\sum_{j=1}^{n} \exp(k_j^\top q)}$ <br/>


- **Create Context Vector using $\large \alpha_{i}$**

    - $\large c_{j}\,=\,\alpha_{1}\,h_{1} + \alpha_{2}\,h_{2} + \ldots + \alpha_{m}\,h_{m}$ = $\large \sum\limits_{i=1}^{m}\,\alpha_{ji}\,h_{i}$


- **Compute Hidden states $\large s_{i}$ of decoder with context vector $\large c_{i}$**

    <img src="https://github.com/SuminizZ/Physics/assets/92680829/b37713b4-25a1-4561-9cb8-2d906bcf66bc" width="400px"> <br/>

    &emsp;&emsp;where $\large C_{i}\,=\, \sum\limits_{j=1}^{m}\,\alpha_{ij}\,h_{j}$







