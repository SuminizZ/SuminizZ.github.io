---
title : "[Information Theory] Understanding Entropy, Cross-Entropy, KL-Divergence"
categories : 
    - Related Topics
tag : [Entropy, deep learning, Cross-Entropy, KL-Divergence]
toc : true
toc_sticky : true
---

## Entropy
- Entropy is a very common concept in physics that acts as a measure of randomness, uncertainty and disorder in a system. Higher entropy means higher randomness.
- This concept can be also applied to information theory, but in a slightly different way<br/>
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/92680829/175291173-3092d2c5-2951-41ea-8a48-09f48169da58.png">
    
- Entropy in information theory describes <span style="color:blue">"Averaged minimum number of bits required to identify information"</span>
- The essential idea of this information theory is that rare events contain more information, whereas common events contain smaller information, thus we need **more resources (bits) to represent rare events**
- This sounds pretty plausible as rare information such as address or phone number can specify someone much better than a very common information such as sex or age.

- But then, how can we **mathmatically measure the amount of information certain event contains** ?
    - We can describe the amount of information of a certain event with respect to the probability of its occurence, which was devised by R. V. Hartley 
        - <img width="180" alt="image" src="https://user-images.githubusercontent.com/92680829/175305691-fc7c5902-5567-4b9f-a33a-0a5133e94a6e.png">
        - <img src="https://user-images.githubusercontent.com/92680829/175309488-29035751-9d20-4059-a7f1-b04c29c54266.png" width="420">
        - According to the graph above, as the rareness of event increases (as the probability of occurence decreases), the amount of information increases as well 
        - For "a" (base of log), we usually set this value as 2 because "bit" (a binary value, either 0 or 1) is commonly used as a basic unit of information

    - **For example**, 
        - Suppose you have a machine that returns A, B, C, D, E, F, G, H with a probability of 0.4, 0.05, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05 respectively (summed as 1). 
        - Then, you need at least 3 bits (2x2x2) to represent all 8 features (A ~ H) 
            - 000 - A / 001 - B / 010 - C / 011 - D / 100 - E / 101 - F / 110 - G / 111 - H
            - This shows that the **amount of bits you need to identify "N" features** can be described as<br/> 
                <img width="70" alt="image" src="https://user-images.githubusercontent.com/92680829/175296381-0d61454a-3052-4a99-851b-14959aa906be.png">
            
       
        - However, previously I've mentioned that each event (A - H) has different occuring frequency and we need to consider the **"rareness"** of an evnet to measure the amount of information it carries (here, entropy) 
            - Then, let's give smaller bit to common events such as A, C (occurrence probability 0.7) and give greater bits to relatively rare events B, D, E, F, G, H (occurrence probability 0.3)
            - We need only 1 bit to define A and C, but need 2 more bits to define the rest of them
            - Therefore, the modified expected value of the number of bits required to carry all information is<br/> <span style="color:blue">0.7 * 1 + 0.3 * 2 = **1.3**</span>, which is quite smaller than the previous value, <span style="color:blue">**3**</span>
    
    <br/>
- To sum up, **Entropy** is the **expected value (mean) of the number of bits to contain information.**<br/>
       - <img width="350" alt="image" src="https://user-images.githubusercontent.com/92680829/175301389-2e17a0fa-d968-4384-b705-705ebf02a3e5.png">
        
        
## Cross-Entropy