---
layout: post
title : "[Paper Review & Implementation] Attention Is All You Need (Transformer, 2017)"
img: papers/transformer/transformer.png
categories: [papers-transformer]  
tag : [Paper Review, Attention, Transformer, PyTorch]
toc : true
toc_sticky : true
---

## Outlines 
- [**Reference**](#reference)
- [**Constraint of Recurrent Models : Sequential Computation**](#constraint-of-recurrent-models--sequential-computation)
- [**Attention of Transformer**](#attention-of-transformer)
- [**Encoder and Decoder Architecture**](#encoder-and-decoder-architecture)
- [**Comparisoin of Computational Efficiency to Other Models**](#comparisoin-of-computational-efficiency-to-other-models)
- [**Performance of Transformer in Machine Translation**](#performance-of-transformer-in-machine-translation)


<br/>

##  **Reference**

- [Attention Is All You Need, Ashish Vaswani, 2017](https://arxiv.org/abs/1706.03762){:target="_blank"}
- [[NLP 논문 구현] pytorch로 구현하는 Transformer (Attention is All You Need)](https://cpm0722.github.io/pytorch-implementation/transformer){:target="_blank"}

<br/>

## **Constraint of Recurrent Models : Sequential Computation** 

<br/>

- The most fundamental weakness of traditional recurrent models (e.g. RNN, LSTM) is that they process the data sequentially. 

- Hidden state ($\large h_{t}$) for every time step depends on the hidden state passed from previous time step ($\large h_{t-1}$) and the current input ($\large x_{t}$).

- This sequential nature of recurrent networks significantly limits their ability to capture long sentences as the amout of computations required increases with the legnth of sentence. 

<br/>

&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/3deba090-4167-4df7-b150-5bfefab58b09" width="500">

<br/>

- Transformer introduced in this paper successfully removes the recurrent characteristics out of the network architecture and only utilizes the attention mechanisms.

- Attention has been a widely used mechanism in NLP as it enables the model to figure out interdependencies between sequences and focus on the particalar part with great relevance to the target position being predicted.

- Using this method, one can compute the relative association between the entire sequences and current target with just a single matrix multiplication.

<br/>

## **Attention of Transformer**

<br/>

### **Scaled Dot-Product Attention**

<br/>

&emsp;&emsp;&emsp;<img width="300" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/6fda8245-3a56-4822-b970-1011ed843e85">

- This is the typical attention where a single set of query, key, and value vectors is used to compute the attention weights between the input elements. 

- Each one of query ($\large Q$), key ($\large K$), and value ($\large V$) is a linear projection of target and input sequences. 
  
  - Query comes from the target sequences while key and value vector is from input sequences.
  
  - Although key and value vectors originate from the same source sequences, they possess distinct values due to undergoing different linear transformation.


- **Attention Score**

  &emsp;&emsp;$\large \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\$

  - Take the dot product between a query and the corresponding key to compute the compatibility between the input elements and target position. 
  
  - Scale it with $\large \sqrt{d_k}$, which is the embedding dimension of a signle attention head. (equals to $\large \frac{d_{model}}{h}$)
    
    - This is because the variation of a single element of attention score (dot product of q and k) increases by a factor of $\large d_k$ compared to that of q and k. 
    
  
  - Now apply softmax function to get an attention probability matrix, which provides a probabilistic representation of how much the input sequences and targe sequences are related.   

- Multiply the attention probability score matrix with the value to finally get an attention.




<br/>

### **Multi-Head Attention (MHA)**

<br/>

&emsp;&emsp;&emsp;<img width="300" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/76dee4a6-5838-427e-b51c-a2c13abbf8af">


- The Transformer introduces a novel modification to the conventional attention mechanism by increasing the number of attention layers running in parallel, referred to as "Multi-Head Attention". 

- This is done by multiplying the number of attention by a factor **h** (named as attention head) and computing the attention score for every head. 

- Intuitively, each of attention head holds a distinct contextual information captured from different parts of input sequences, which increases expressiveness and provides a more comprehensive "attention" with respect to the target word. 

- **Attention Scores**

  &emsp;&emsp;$\large \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \text{WO}, \text{where } \text{head}_i = \text{Attention}(QWQ_i, KW K_i, V WV_i)\$

  - All the underlying computations are same as the scaled dot-product attention explained above, but the operation is repeated by the number of heads. 

  - In the final stage of multi-head attention layer, every attention computed in parallel is cocatenated and enters into the **point-wise feed forward layer**.

<br/>


### **Point-Wise Feed Forward Layer**

<br/>

&emsp;&emsp;$\large \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2\$

- FC layer -> ReLU (GELU these days) non-linearity -> FC layer 

- Can add complexity and expressiveness of each features computed from attention layers.

<br/>


## **Embedding and Positional Encoding**

<br/>

### **Token Embedding**

<br/>

&emsp;&emsp;&emsp;<img width="316" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/be918008-9f8e-41b4-9e7e-5971e9ce8a4c">

<br/>

- As always applied to NLP datasets, each token of input sequences is embedded into a certain dimensionality.

- Parameters of embedding layer are learned during training.

<br/>

### **Positional Encoding**

<br/>

- Although parallelized operation of transformer is a huge computational advantage over recurrent networks, it leads to the loss of positional information of sequential data, which is critical in NLP. 

- To give an information about the relative order of the sequence to transformer networks, authors add a pre-processing step called "positional encdoing".

<br/>

#### **Positional Encoding Matrix**

<br/>

- $\large P \in \mathbb{R}^{l\times d}$, where $P_{ij} = $ $$\large \begin{cases}
\text{sin}\left(i \cdot 10000^{-\frac{j}{d}}\right) & \text{if j is even} \\
\text{cos}\left(i \cdot 10000^{-\frac{(j-1)}{d}}\right) & \text{otherwise} \\
\end{cases}$$

- Here, $\large i$ indicates the $\large i\text{th}$ position of token, and $\large j$ indicates $\large j\text{th}$ dimension of entire embedding dimension.

<br/>

&emsp;&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/148128a4-0a8e-4612-97b1-a3287e5d7411" width="600">


- Each position $\large i$ in input sequence corresponds to a sinusoid with distinct wavelength, $\large \frac{2π}{10000^{-j/d}}$

<br/>

&emsp;<img src="https://github.com/SuminizZ/Physics/assets/92680829/4ba11c2c-6b42-482a-ac3d-395e13a9b911" width="700">

- $\large k$ here corresponds to $\large i$.


<br/>


## **Encoder and Decoder Architecture**

<br/>

&emsp;&emsp;&emsp;<img width="500" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/e434073b-0c93-45a2-b21b-222399ac03ce">

<br/>

- Transformer in this paper is a **seq-to-seq model** that performs translation tasks.

- Input sequence for encoder would be a in source language to be translated and one of the inputs and the output of the decoder is in translated target language. 

- Encoder transfers the fixed-dimensional encoded sequences, often referred to as **Context Vector**, to decoder and decoder utilizes it as key and value vectors for multi-head attention (not self)

<br/>

### **Encoder**

<br/>

- **Self Multi-Head Attention**
  
  - Query, key, and value vectors all are from input sequences, computing attention with respect to itself.
  
- **Encoder Block**

  - Single encoder block consistis of two sub-layers, a multi-head attention layer followed by layer normalization and a point-wise feed forward layer also followed by layer norm. 

  - Repaeat a block for **N** times to get final output.

  - Make sure that the shapes of input and output of every layer and block are identical. 

- **Residual Connection**
  
  - Residual connection introduced in ResNet architecture is also adopted here in transformer around each of the two sub layers (multi-head attention and point-wise feed forward).

- Provides the context vectors for Decoder.

<br/>

### **Decoder**

<br/>

- Composed of a stack of N identical layers with three sub-layers
  
  - A modification added to prevent the model from getting access to positions subsequent to current position (Explained later in **Subsequent Masking**)

- Performs two multi-head attentions: one for self-attention, which operates on the target input sequences, and one for cross-attention, which connects the output of the first attention layer of the decoder with the encoded source sequences transferrred from the encoder.

- Output of second attention layer is passed feed-forward layer 

- Residual connections around each of the sub-layers, followed by layer normalization.


<br/>

### **Pad Masking and Subsequent Masking**

<br/>

- **Pad Mask**
  
  - All input sentences are padded to have fixed length. 
  
  - Padded sequences should not be considered when computing attention scores, thus need to be masked. 

- **Subsequent Mask**

  - During training, the target input sequences entered into decoder are not it's own predictions. Instead, it uses the method called **"Teacher Forcing"** where ground-truth translated sequences are given as an input to the decoder. 

  - This is because in the machine translation task, the model outputs the next word based on the previous output generated by the model. During the initial stage of learning, it is highly unlikely for the model to give correct predictions as the parameters have not been sufficiently optimized yet. Hence, training the model given the incorrect inputs accumulates the errors and thus significantly drops the speed of convergence.  

  - To tackle this issue, one gives the model an access to correct ground-truth to stablilize and speed up the training process. 

  - During the **inference phase**, the initial input sequences to the decoder is typically a <sos> (start-of-sequence) token followed by generated elements of the output sequences. 
    
<br/>

## **Comparisoin of Computational Efficiency to Other Models**

<br/>

<img width="780" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/b9b17f6c-1054-4adb-80ea-7ef1e011f40e">

<br/>

## **Performance of Transformer in Machine Translation**

<br/>

<img width="700" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/8641ae83-5d4d-460c-b195-d621b7c228b2">

<br/>

### **Model Architectures**

<br/>

<img width="750" alt="image" src="https://github.com/SuminizZ/Algorithm/assets/92680829/cb0b9f68-3f82-469e-9b78-e83e4e443931">