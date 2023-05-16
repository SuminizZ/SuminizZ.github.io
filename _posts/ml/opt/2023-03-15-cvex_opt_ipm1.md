---
layout: post
title : "[Convex Optimization] Interior-Point Method with Barrier Function"
img: ml/opt/ipm_b.png
categories: [ml-opt] 
tag : [Optimization, ML]
toc : true
toc_sticky : true
---

<br>

- Optimization algorithm commonly used to solve convex optimization problems using Newton method 
- starts with perturbed KKT by introducing a perturbation variable to complementary slackness condition
- involves introducing a barrier function into the objective function to create a modified problem that is easier to solve. 
    - barrier function acts as a barrier that prevents the algorithm from reaching points outside of the feasible region. (inequality boundary)
    - typically a logarithmic function that approaches infinity as the solution approaches the equality boundary 

- solves equality constrained optimization problem by iteratively improving the solution untill the improvement is below certain threshold

### Summary Notes

- [<span style="color:purple">**Interior-Point Method with Barrier**</span>](https://drive.google.com/file/d/17ba9-kTX6MrGtiqev28PU8YCRVXumk01/view?usp=share_link){:target="_blank"}


