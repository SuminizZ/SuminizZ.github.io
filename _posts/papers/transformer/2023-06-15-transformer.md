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
- [**Issue of Interest**](#issue-of-interest)
- [**Model Architectures of BiRNN with Attention**](#model-architectures-of-birnn-with-attention)
  * [**1. Encoder**](#1.-encoder)
  * [**2. Decoder**](#2.-decoder)
- [**BiRNN Ecoder-Decoder with Attention Mechanism Summary**](#birnn-ecoder-decoder-with-attention-mechanism-summary)

<br/>

##  **Reference**

<br/>

- [Attention Is All You Need, Ashish Vaswani, 2017](https://arxiv.org/abs/1706.03762){:target="_blank"}
- [[NLP 논문 구현] pytorch로 구현하는 Transformer (Attention is All You Need)](https://cpm0722.github.io/pytorch-implementation/transformer){:target="_blank"}

<br/>

## **Constraint of Recurrent Models : Sequential Computation** 

<br/>

- The most fundamental weakness of traditional recurrent models (e.g. RNN, LSTM) is that they process the data sequentially. 
- Hidden state ($\large h_{t}$) for every time step depends on the hidden state passed from previous time step ($\large h_{t-1}$) and the current input ($\large x_{t}$), 
- This sequential nature of recurrent networks significantly limits their ability to capture long sentences as the amout of computations required increases with the legnth of sentence. 

<br/>

&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/3deba090-4167-4df7-b150-5bfefab58b09" width="500">

<br/>

- Transformer introduced in this paper successfully removes the recurrent part out of the network architecture and only utilizes the attention mechanisms.

<br/>

## **Multi-Head Attention**

<br/>

- Transformer adds a novel modific

not only just adopts the standard attention mechanisms that extract only one 

