---
title: "From Doppler to Gravity: A Physical Analogy for Transformer Positional Encoding"
date: 2026-03-27 23:00:00 +0800
categories: [Deep Learning, Physics]
tags: [transformer, positional encoding, rope, general relativity, world models]
math: true
toc: true
image:
  path: /assets/img/posts/physics/black_hole.jpg
  alt: black hole
---

## Abstract

Transformer models rely on positional encodings to inject sequence order. This post explores a fascinating connection between these encodings and physical wave phenomena. We first show that sinusoidal positional encoding shares the same mathematical structure as the Doppler effect—both rely on a linear relationship between position/time and phase. Building on this, we draw inspiration from gravitational redshift in general relativity to propose a *curved spacetime positional encoding*, where different regions of a sequence can have different “temporal resolutions” depending on their semantic importance. This perspective offers a path toward more adaptive, physically grounded world models.

---

## 1. Introduction

The attention mechanism in Transformers is permutation-invariant by design, so positional information must be injected explicitly. The original Transformer used sinusoidal absolute position encodings, while later works like RoPE (Rotary Position Embedding) elegantly encode relative positions via rotations. However, these methods assume a *uniform* sequence space—each position is treated equally, with the same “flow of time”.

Meanwhile, physics gives us the Doppler effect (frequency shifts due to relative motion) and gravitational redshift (frequency shifts due to spacetime curvature). Both describe how wave phases accumulate linearly with coordinates, but the rate can vary with context. This raises an intriguing question: **can we design a positional encoding that adapts its “clock speed” based on the importance of each token, much like how gravity slows down time near massive objects?**

In this post, we explore this analogy step by step.

---

## 2. Positional Encoding and the Doppler Effect
![img-description](/assets/img/posts/physics/gravity.jpg)
### 2.1 Sinusoidal Encoding: A Frequency Perspective

The original Transformer uses sine and cosine functions of different frequencies:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Define the base frequency $\omega_i = 1 / 10000^{2i/d}$. Then the encoding vector becomes:

$$
\mathbf{PE}(pos) = \big[\sin(\omega_1 pos),\; \cos(\omega_1 pos),\; \sin(\omega_2 pos),\; \dots\big]
$$

The core property is a **linear relationship** between position $pos$ and the phase $\theta = \omega_i \cdot pos$. This linear phase accumulation allows the model to express relative positions via trigonometric identities.

### 2.2 Doppler Effect in Wave Terms

In the classical Doppler effect (for sound), the observed frequency $f'$ relates to the source frequency $f_0$ by:

$$
f' = f_0 \cdot \frac{c}{c \pm v_s}
$$

If we write the observed wave as $\sin(2\pi f' t + \phi_0)$, the phase $\phi(t) = 2\pi f' t + \phi_0$ also grows **linearly with time** $t$. Identifying time $t$ with the position index $pos$, the Doppler effect is mathematically analogous: the rate of phase accumulation (frequency) is modulated by relative motion, just as the frequencies $\omega_i$ in sinusoidal encoding are fixed.

### 2.3 RoPE: From Absolute Phase to Relative Rotation

RoPE goes a step further by directly embedding positions into the query and key vectors via rotation matrices. For a position $m$, the rotation matrix $R_{\Theta,m}$ acts on the query $\mathbf{q}_m$ and key $\mathbf{k}_n$ such that:

$$
\langle R_{\Theta,m} \mathbf{q}_m,\; R_{\Theta,n} \mathbf{k}_n \rangle = \langle \mathbf{q}_m,\; R_{\Theta,n-m} \mathbf{k}_n \rangle
$$

Here the rotation angle for dimension $i$ is $\theta_i m$, again linear in position. This is reminiscent of Doppler radar, where the phase difference encodes relative velocity/displacement.

---

## 3. Gravitational Redshift: Toward Non‑Uniform Positional Encoding

### 3.1 Time Dilation and Frequency Redshift

In general relativity, a clock in a stronger gravitational field ticks slower. For an observer far from the source, light emitted from a strong gravity region is redshifted:

$$
f_{\text{obs}} = f_{\text{local}} \cdot \sqrt{1 - \frac{2GM}{rc^2}}
$$

The local frequency is lower when the gravitational potential is deeper. In other words, **the local “pace of time” modulates the effective frequency**.

### 3.2 Curved Spacetime Positional Encoding

If we view RoPE as a “flat spacetime” encoding with uniform rotation rates, then gravitational redshift suggests a natural generalization: let the rotation speed vary with the **semantic importance** of each token. Important tokens (high information density) act like massive objects that slow down the local rotation—i.e., they stretch time.

We introduce a **learnable gravitational potential** $\Phi_m$ for each position $m$, derived from the token’s content. The angular increment for dimension $i$ at step $m$ becomes:

$$
\Delta \phi_i(m) = \theta_i \cdot \gamma(\Phi_m)
$$

where $\gamma(\Phi)$ is a redshift factor (monotonically decreasing, e.g., $\gamma(\Phi)=e^{-\beta\Phi}$). The cumulative phase is then:

$$
\phi_i(m) = \sum_{k=0}^{m-1} \Delta \phi_i(k)
$$

The rotation matrix $R(\phi_i(m))$ is applied to the query/key vectors as before. This is fully differentiable, and the potential $\Phi_m$ can be learned from the input.

---

## 4. Proposed Framework

### 4.1 Learning the Gravitational Potential

Let the input sequence be $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_L]$. A small network computes the potential:

$$
\Phi_m = \sigma(\mathbf{W}_\Phi \mathbf{x}_m + b_\Phi)
$$

where $\sigma$ is an activation (e.g., sigmoid) to keep $\Phi_m$ bounded. $\Phi_m$ can be interpreted as the “semantic weight” or “information density” of token $m$.

### 4.2 Redshift‑Modulated Rotation

Given a hyperparameter $\beta$ controlling redshift strength:

$$
\Delta \phi_i(m) = \theta_i \cdot \exp(-\beta \Phi_m)
$$

The cumulative phase is obtained via a prefix sum:

$$
\phi_i(m) = \sum_{k=1}^{m} \Delta \phi_i(k)
$$

Finally, for each dimension pair $(2i, 2i+1)$, we apply the 2D rotation:

$$
\begin{bmatrix}
q_m^{(2i)} \\ q_m^{(2i+1)}
\end{bmatrix}_{\text{new}} = 
\begin{bmatrix}
\cos\phi_i(m) & -\sin\phi_i(m) \\
\sin\phi_i(m) & \cos\phi_i(m)
\end{bmatrix}
\begin{bmatrix}
q_m^{(2i)} \\ q_m^{(2i+1)}
\end{bmatrix}
$$

and similarly for keys.

### 4.3 Attention Computation

The attention score between positions $m$ and $n$ now depends on the path‑integrated phases, not just on the difference $m-n$. This allows the model to learn **asymmetric, context‑dependent positional relationships**: two pairs with the same absolute distance can have different effective phase differences if the region between them has high or low potential.

---

## 5. Potential Advantages and Challenges

### 5.1 Expected Benefits

- **Multi‑scale temporal modeling**: Important regions are processed with higher resolution (slower rotation), while unimportant parts are skipped over.
- **Improved long‑range dependencies**: By slowing down rotation in critical segments, the model can keep related positions from drifting too far apart in the complex plane.
- **Physical inductive bias**: The design mimics how real‑world systems handle non‑uniform time scales.
- **Interpretability**: The learned potential $\Phi_m$ can be visualized to reveal which tokens the model deems “heavy” or important.

### 5.2 Challenges to Address

- **Computational cost**: The cumulative sum requires sequential scanning, but can be parallelized with prefix‑sum algorithms or approximated.
- **Training stability**: Large variations in $\Phi_m$ may cause unstable gradients; regularization techniques will be needed.
- **Integration with efficient attention**: Adapting to FlashAttention or other optimized kernels may require careful implementation.

---

## 6. Future Directions

### 6.1 Relativistic Attention

A more ambitious extension is to incorporate the spacetime metric directly into the attention score:

$$
\text{Attention}(\mathbf{q}_m,\mathbf{k}_n) = \text{softmax}\!\left( \frac{\mathbf{q}_m^T\mathbf{k}_n}{\sqrt{d}} + \text{metric}(\Phi_m,\Phi_n,m,n) \right)
$$

where the metric term could encode the interaction between potentials.

### 6.2 Multi‑Body Gravity

In many sequences, different entities (e.g., multiple objects in a scene, multiple characters in a story) act as independent “gravity centers”. One can define a potential as a superposition:

$$
\Phi_m = \sum_{c} \Phi_c(m)
$$

where each $\Phi_c$ is centered around a specific entity or concept.

### 6.3 Connection to Neuroscience

The grid cells and place cells in the hippocampus exhibit non‑uniform spatial coding that scales with environmental features. This curved positional encoding might offer a computational model for such neural representations.

---

## 7. Conclusion

We have traced an intriguing line of thought: from the linear‑phase accumulation of sinusoidal positional encodings (analogous to the Doppler effect), to the uniform rotation of RoPE, and finally to a generalization inspired by gravitational redshift. The resulting **curved spacetime positional encoding** allows a Transformer to adaptively allocate different “temporal resolutions” to different parts of a sequence, based on learnable semantic potentials.

This cross‑disciplinary analogy not only deepens our understanding of existing positional encoding mechanisms but also opens the door to more flexible, physically‑informed architectures. We hope this perspective inspires further exploration at the intersection of deep learning and fundamental physics.

---

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
3. Doppler, C. (1842). Über das farbige Licht der Doppelsterne.
4. Einstein, A. (1915). Die Feldgleichungen der Gravitation.
5. Tay, Y., et al. (2021). Long Range Arena: A Benchmark for Efficient Transformers. *ICLR*.
6. O’Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. *Brain Research*.
7. Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*.

---

## Appendix: Conversation History

> **User:** transformer中的位置编码和多普勒效应的联系  
> **Assistant:** (Detailed response connecting positional encoding with Doppler effect, covering mathematical homomorphism, physical intuition, and RoPE.)

> **User:** 更进一步，如果考虑引力：空间中有强引力、弱引力区域，强引力区域的时间慢，表现出低频，弱引力区域时间快，表现出高频，从这个角度是不是可以做出比现有的transformer + rope更好的结构？呈现出一种对于关键区域的“slow motion”，做出更好的世界模型？  
> **Assistant:** (Response extending the analogy to gravitational redshift, proposing a curved spacetime positional encoding with learnable potential, outlining design, advantages, and future directions.)

> **User:** 将上面的讨论总结成一篇学术文章  
> **Assistant:** (Provided a structured academic article summarizing the discussion with sections: abstract, introduction, Doppler analogy, gravitational extension, framework, experiments, discussion, references.)
